use learn_agent_rs::agent::{Agent, AgentConfig};
use learn_agent_rs::tools::{ToolRegistry, default_tool_registry};
use learn_agent_rs::types::{
    AssistantMessage, ChatResponse, FunctionCall, Message, Role, ToolCall,
};

// ─── Types tests ─────────────────────────────────────────────────────────────

#[test]
fn test_message_user() {
    let msg = Message::user("Hello");
    assert_eq!(msg.role, Role::User);
    assert_eq!(msg.content.as_deref(), Some("Hello"));
    assert!(msg.tool_calls.is_none());
    assert!(msg.tool_call_id.is_none());
}

#[test]
fn test_message_system() {
    let msg = Message::system("Be helpful");
    assert_eq!(msg.role, Role::System);
    assert_eq!(msg.content.as_deref(), Some("Be helpful"));
}

#[test]
fn test_message_tool_result() {
    let msg = Message::tool_result("call_123", "result data");
    assert_eq!(msg.role, Role::Tool);
    assert_eq!(msg.content.as_deref(), Some("result data"));
    assert_eq!(msg.tool_call_id.as_deref(), Some("call_123"));
}

#[test]
fn test_message_from_assistant_with_content() {
    let assistant_msg = AssistantMessage {
        role: Role::Assistant,
        content: Some("Hello user".to_string()),
        tool_calls: None,
    };
    let msg = Message::from_assistant(&assistant_msg);
    assert_eq!(msg.role, Role::Assistant);
    assert_eq!(msg.content.as_deref(), Some("Hello user"));
    assert!(msg.tool_calls.is_none());
}

#[test]
fn test_message_from_assistant_with_tool_calls() {
    let tool_calls = vec![ToolCall {
        id: "tc_1".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"Paris"}"#.to_string(),
        },
    }];
    let assistant_msg = AssistantMessage {
        role: Role::Assistant,
        content: None,
        tool_calls: Some(tool_calls),
    };
    let msg = Message::from_assistant(&assistant_msg);
    assert_eq!(msg.role, Role::Assistant);
    assert!(msg.content.is_none());
    assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
    assert_eq!(
        msg.tool_calls.as_ref().unwrap()[0].function.name,
        "get_weather"
    );
}

// ─── Serialization round-trip tests ──────────────────────────────────────────

#[test]
fn test_message_serialization_round_trip() {
    let msg = Message::user("test message");
    let json = serde_json::to_string(&msg).unwrap();
    let deserialized: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(msg, deserialized);
}

#[test]
fn test_tool_call_serialization() {
    let tc = ToolCall {
        id: "call_abc".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "calculate".to_string(),
            arguments: r#"{"expression":"1+1"}"#.to_string(),
        },
    };
    let json = serde_json::to_string(&tc).unwrap();
    assert!(json.contains("\"type\":\"function\""));
    assert!(json.contains("\"name\":\"calculate\""));

    let deserialized: ToolCall = serde_json::from_str(&json).unwrap();
    assert_eq!(tc, deserialized);
}

#[test]
fn test_chat_response_deserialization() {
    let json = r#"{
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }"#;

    let response: ChatResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.id, "chatcmpl-123");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("Hello!")
    );
    assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
    assert_eq!(response.usage.as_ref().unwrap().total_tokens, 15);
}

#[test]
fn test_chat_response_with_tool_calls_deserialization() {
    let json = r#"{
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1700000001,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_xyz",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"London\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": null
    }"#;

    let response: ChatResponse = serde_json::from_str(json).unwrap();
    assert!(response.choices[0].message.content.is_none());
    let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
}

// ─── Tool registry tests ────────────────────────────────────────────────────

#[test]
fn test_default_registry_has_all_tools() {
    let registry = default_tool_registry();
    let defs = registry.definitions();
    let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
    assert!(names.contains(&"get_weather"));
    assert!(names.contains(&"calculate"));
    assert!(names.contains(&"read_file"));
}

#[test]
fn test_tool_definitions_have_correct_type() {
    let registry = default_tool_registry();
    for def in registry.definitions() {
        assert_eq!(def.tool_type, "function");
    }
}

#[test]
fn test_get_weather_returns_valid_json() {
    let registry = default_tool_registry();
    let result = registry
        .execute("get_weather", r#"{"location":"New York"}"#)
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["location"], "New York");
    assert!(parsed["temperature_celsius"].is_number());
    assert!(parsed["condition"].is_string());
    assert!(parsed["humidity_percent"].is_number());
}

#[test]
fn test_calculate_basic_operations() {
    let registry = default_tool_registry();

    let result = registry
        .execute("calculate", r#"{"expression":"7 + 3"}"#)
        .unwrap();
    assert_eq!(result, "10");

    let result = registry
        .execute("calculate", r#"{"expression":"20 - 8"}"#)
        .unwrap();
    assert_eq!(result, "12");

    let result = registry
        .execute("calculate", r#"{"expression":"6 * 7"}"#)
        .unwrap();
    assert_eq!(result, "42");

    let result = registry
        .execute("calculate", r#"{"expression":"15 / 4"}"#)
        .unwrap();
    assert_eq!(result, "3.75");
}

#[test]
fn test_calculate_error_on_division_by_zero() {
    let registry = default_tool_registry();
    let result = registry.execute("calculate", r#"{"expression":"10 / 0"}"#);
    assert!(result.is_err());
}

#[test]
fn test_read_file_with_existing_file() {
    // Read Cargo.toml which we know exists
    let registry = default_tool_registry();
    let result = registry
        .execute("read_file", r#"{"path":"Cargo.toml"}"#)
        .unwrap();
    assert!(result.contains("[package]"));
    assert!(result.contains("learn-agent-rs"));
}

#[test]
fn test_custom_tool_registry() {
    let mut registry = ToolRegistry::new();
    registry.register(
        "echo",
        "Echo back the input",
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        }),
        |args| {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
            Ok(format!("Echo: {text}"))
        },
    );

    assert_eq!(registry.definitions().len(), 1);
    let result = registry.execute("echo", r#"{"text":"hello"}"#).unwrap();
    assert_eq!(result, "Echo: hello");
}

// ─── Agent construction tests ────────────────────────────────────────────────

#[test]
fn test_agent_construction_no_system_prompt() {
    let config = AgentConfig {
        model: "test".to_string(),
        url: "http://localhost:8080/v1/chat/completions".to_string(),
        api_key: "key".to_string(),
        max_iterations: 10,
    };
    let agent = Agent::new(config, None);
    assert!(agent.history.is_empty());
}

#[test]
fn test_agent_construction_with_system_prompt() {
    let config = AgentConfig {
        model: "test".to_string(),
        url: "http://localhost:8080/v1/chat/completions".to_string(),
        api_key: "key".to_string(),
        max_iterations: 10,
    };
    let agent = Agent::new(config, Some("System prompt"));
    assert_eq!(agent.history.len(), 1);
    assert_eq!(agent.history[0].role, Role::System);
    assert_eq!(agent.history[0].content.as_deref(), Some("System prompt"));
}

// ─── Request serialization tests ─────────────────────────────────────────────

#[test]
fn test_chat_request_serialization_without_tools() {
    use learn_agent_rs::types::ChatRequest;

    let request = ChatRequest {
        model: "gpt-4".to_string(),
        messages: vec![Message::user("Hi")],
        tools: None,
    };

    let json = serde_json::to_string(&request).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["model"], "gpt-4");
    assert!(parsed["messages"].is_array());
    // tools field should not be present when None
    assert!(parsed.get("tools").is_none());
}

#[test]
fn test_chat_request_serialization_with_tools() {
    use learn_agent_rs::types::ChatRequest;

    let registry = default_tool_registry();
    let request = ChatRequest {
        model: "gpt-4".to_string(),
        messages: vec![Message::system("Be helpful"), Message::user("Hello")],
        tools: Some(registry.definitions().to_vec()),
    };

    let json = serde_json::to_string(&request).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert!(parsed["tools"].is_array());
    assert_eq!(parsed["tools"].as_array().unwrap().len(), 3);
    assert_eq!(parsed["messages"].as_array().unwrap().len(), 2);
}

// ─── Role serialization tests ────────────────────────────────────────────────

#[test]
fn test_role_serialization() {
    assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
    assert_eq!(
        serde_json::to_string(&Role::Assistant).unwrap(),
        "\"assistant\""
    );
    assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
    assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
}

#[test]
fn test_role_deserialization() {
    assert_eq!(
        serde_json::from_str::<Role>("\"user\"").unwrap(),
        Role::User
    );
    assert_eq!(
        serde_json::from_str::<Role>("\"assistant\"").unwrap(),
        Role::Assistant
    );
    assert_eq!(
        serde_json::from_str::<Role>("\"system\"").unwrap(),
        Role::System
    );
    assert_eq!(
        serde_json::from_str::<Role>("\"tool\"").unwrap(),
        Role::Tool
    );
}

// ─── Message skip_serializing_if tests ───────────────────────────────────────

#[test]
fn test_user_message_omits_optional_fields() {
    let msg = Message::user("test");
    let json = serde_json::to_string(&msg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    // Should have role and content only
    assert!(parsed.get("role").is_some());
    assert!(parsed.get("content").is_some());
    // tool_calls and tool_call_id should be absent (not null)
    assert!(parsed.get("tool_calls").is_none());
    assert!(parsed.get("tool_call_id").is_none());
}

#[test]
fn test_tool_result_message_includes_tool_call_id() {
    let msg = Message::tool_result("call_1", "result");
    let json = serde_json::to_string(&msg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["role"], "tool");
    assert_eq!(parsed["tool_call_id"], "call_1");
    assert_eq!(parsed["content"], "result");
}
