#!/usr/bin/env python3
import argparse
import json
import time
import os
import sys
import logging
from datetime import datetime

import cv2
from ultralytics import YOLO

# SDK do Greengrass para IPC
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import QOS, ServiceError

# --- Configuração de logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Função para publicar detecções ---
def publish_detection(ipc_client: GreengrassCoreIPCClientV2, topic: str, payload: dict):
    try:
        payload_json = json.dumps(payload).encode("utf-8")
        ipc_client.publish_to_iot_core(
            topic_name=topic,
            qos=QOS.AT_LEAST_ONCE,
            payload=payload_json
        )
        logger.info(f"Mensagem publicada com sucesso no tópico: {topic}")
    except ServiceError as e:
        logger.error(f"Erro ao publicar no IoT Core: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Erro inesperado durante a publicação: {e}", exc_info=True)

# --- Função principal ---
def main():
    parser = argparse.ArgumentParser(description="Detector de Veículos YOLOv8 para AWS Greengrass")
    parser.add_argument("--source", required=True, help="Arquivo de vídeo")
    parser.add_argument("--topic", required=True, help="Tópico MQTT para publicar detecções")
    parser.add_argument("--conf", type=float, default=0.50, help="Confiança mínima para detecção")
    parser.add_argument("--labels", required=True, help="Classes a serem detectadas, separadas por vírgula")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "models", "yolov8n.pt"), help="Caminho do modelo YOLOv8")

    args = parser.parse_args()

    # --- Configura cache gravável para gravar resultados---
    cache_dir = "/greengrass/v2/work/com.example.VehicleDetector/cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["ULTRALYTICS_CACHE"] = cache_dir

    logger.info("--- Iniciando VehicleDetector ---")
    logger.info(f"Fonte do vídeo: {args.source}")
    logger.info(f"Tópico MQTT: {args.topic}")
    logger.info(f"Confiança mínima: {args.conf}")
    logger.info(f"Modelo YOLO: {args.model}")
    logger.info(f"Cache dir: {cache_dir}")

    allowed_labels = {s.strip() for s in args.labels.split(",") if s.strip()}
    logger.info(f"Classes permitidas: {allowed_labels}")

    # --- Carrega modelo YOLO ---
    try:
        model = YOLO(args.model)
    except Exception as e:
        logger.error(f"Falha ao carregar modelo YOLO: {e}", exc_info=True)
        sys.exit(1)

    # --- Inicializa captura de vídeo ---
    cap_source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        logger.error(f"Não foi possível abrir a fonte de vídeo: '{args.source}'")
        sys.exit(1)

    # --- Inicializa IPC do Greengrass ---
    ipc_client = None
    for i in range(5):
        try:
            ipc_client = GreengrassCoreIPCClientV2()	
            logger.info("Cliente IPC conectado com sucesso ao Nucleus do Greengrass.")
            break
        except Exception as e:
            logger.warning(f"Falha ao conectar IPC, retry em 5s ({i+1}/5)...")
            time.sleep(5)
    if ipc_client is None:
        logger.error("Não foi possível conectar ao Greengrass IPC")
        sys.exit(1)

    # --- Loop principal ---
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("Fim do vídeo ou fonte indisponível. Reiniciando em 10s.")
                cap.release()
                time.sleep(10)
                cap = cv2.VideoCapture(cap_source)
                if not cap.isOpened():
                    logger.error("Falha ao reabrir a fonte de vídeo. Encerrando.")
                    break
                continue

            # --- Inferência ---
            t_start = time.time()
            results = model.predict(source=frame, verbose=False, conf=args.conf, imgsz=640)[0]
            t_latency = int((time.time() - t_start) * 1000)

            # --- Processa detecções ---
            detections = []
            for box in results.boxes:
                class_id = int(box.cls.item())
                class_name = results.names[class_id]
                if class_name in allowed_labels:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({
                        "class": class_name,
                        "confidence": round(float(box.conf.item()), 3),
                        "bbox": [x1, y1, x2, y2]
                    })

            # --- Publica resultados ---
			
            message_payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": args.source,
                "inference_latency_ms": t_latency,
                "detection_count": len(detections),
                "detections": detections
            }

            publish_detection(ipc_client, args.topic, message_payload)

    except KeyboardInterrupt:
        logger.info("Recebido sinal de interrupção. Encerrando...")
    except Exception as e:
        logger.error(f"Erro inesperado no loop principal: {e}", exc_info=True)
    finally:
        cap.release()
        logger.info("--- VehicleDetector finalizado ---")

if __name__ == "__main__":
    main()
