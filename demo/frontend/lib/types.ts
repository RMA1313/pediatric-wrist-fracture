export type ModelInfo = {
  id: "yolov8" | "yolov9" | "yolo26"
  display_name: string
  english_name: string
  family: string
  checkpoint_available: boolean
  default: boolean
  description?: string | null
}

export type BoxResponse = {
  class_id: number
  class_name: string
  class_name_fa: string
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
  nx1: number
  ny1: number
  nx2: number
  ny2: number
}

export type PredictResponse = {
  request_id: string
  model_id: string
  model_display_name: string
  fracture_detected: boolean
  num_detections: number
  maximum_confidence: number
  inference_time_ms: number
  total_processing_time_ms: number
  original_width: number
  original_height: number
  detections: BoxResponse[]
  annotated_image_url: string
  download_url: string
}

