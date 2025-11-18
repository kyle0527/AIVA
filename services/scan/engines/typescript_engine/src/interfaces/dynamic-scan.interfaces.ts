/**
 * Enhanced Dynamic Scanning Interfaces
 * 增強動態掃描介面定義
 */

export interface DynamicScanTask {
  task_id: string;
  scan_id: string;
  url: string;
  extraction_config: ExtractionConfig;
  interaction_config: InteractionConfig;
  timeout_ms?: number;
}

export interface ExtractionConfig {
  extract_forms: boolean;
  extract_links: boolean;
  extract_ajax: boolean;
  extract_api_calls: boolean;
  extract_websockets: boolean;
  extract_js_variables: boolean;
  extract_event_listeners: boolean;
  wait_for_network_idle: boolean;
  network_idle_timeout_ms: number;
}

export interface InteractionConfig {
  click_buttons: boolean;
  fill_forms: boolean;
  scroll_pages: boolean;
  hover_elements: boolean;
  trigger_events: boolean;
  wait_time_ms: number;
  max_interactions: number;
}

export interface NetworkRequest {
  url: string;
  method: string;
  headers: Record<string, string>;
  post_data?: string;
  response_status?: number;
  response_headers?: Record<string, string>;
  timestamp: number;
}

export interface DOMChange {
  type: 'childList' | 'attributes' | 'subtree';
  target_node: string;
  added_nodes?: string[];
  removed_nodes?: string[];
  attribute_name?: string;
  old_value?: string;
  new_value?: string;
  timestamp: number;
}

export interface InteractionResult {
  interaction_type: 'click' | 'input' | 'hover' | 'scroll' | 'keyboard';
  target_selector: string;
  success: boolean;
  error_message?: string;
  dom_changes: DOMChange[];
  network_requests: NetworkRequest[];
  timestamp: number;
}

export interface DynamicContent {
  content_id: string;
  content_type: 'form' | 'link' | 'api_endpoint' | 'websocket' | 'js_variable' | 'event_listener';
  url: string;
  source_url: string;
  text_content?: string;
  attributes: Record<string, any>;
  confidence: number;
  extraction_method: string;
}

export interface DynamicScanResult {
  task_id: string;
  scan_id: string;
  url: string;
  status: 'completed' | 'failed' | 'timeout';
  contents: DynamicContent[];
  interactions: InteractionResult[];
  network_requests: NetworkRequest[];
  dom_changes: DOMChange[];
  metadata: {
    total_interactions: number;
    total_network_requests: number;
    total_dom_changes: number;
    scan_duration_ms: number;
    javascript_errors: string[];
  };
  error_message?: string;
}

export interface JSVariable {
  name: string;
  value: any;
  type: string;
  scope: 'global' | 'local';
  source_line?: number;
}

export interface EventListener {
  event_type: string;
  target_selector: string;
  handler_code: string;
  source_line?: number;
}