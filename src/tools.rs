use crate::types::{FunctionDefinition, ToolDefinition};
use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::collections::HashMap;

/// A boxed tool handler function.
type ToolHandler = Box<dyn Fn(&Value) -> Result<String> + Send + Sync>;

/// Registry of available tools that the agent can invoke.
pub struct ToolRegistry {
    definitions: Vec<ToolDefinition>,
    /// Maps function name → handler.
    handlers: HashMap<String, ToolHandler>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
            handlers: HashMap::new(),
        }
    }

    /// Register a tool with its definition and handler function.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        handler: impl Fn(&Value) -> Result<String> + Send + Sync + 'static,
    ) {
        let name = name.into();
        self.definitions.push(ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.clone(),
                description: description.into(),
                parameters,
            },
        });
        self.handlers.insert(name, Box::new(handler));
    }

    /// Return tool definitions suitable for the API request.
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// Execute a tool by name with JSON arguments.
    pub fn execute(&self, name: &str, arguments: &str) -> Result<String> {
        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: {name}"))?;
        let args: Value = serde_json::from_str(arguments)
            .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
        handler(&args)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Built-in tool implementations ──────────────────────────────────────────

/// Create a registry populated with default example tools.
pub fn default_tool_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // 1) get_weather — returns mock weather data
    registry.register(
        "get_weather",
        "Get the current weather for a given location. Returns temperature and conditions.",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. \"San Francisco, CA\""
                }
            },
            "required": ["location"]
        }),
        handle_get_weather,
    );

    // 2) calculate — evaluate a simple arithmetic expression
    registry.register(
        "calculate",
        "Evaluate a simple arithmetic expression and return the result. Supports +, -, *, / on integers and floats.",
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. \"2 + 3 * 4\""
                }
            },
            "required": ["expression"]
        }),
        handle_calculate,
    );

    // 3) read_file — read a file from the local filesystem
    registry.register(
        "read_file",
        "Read the contents of a file at the given path and return it as a string.",
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }),
        handle_read_file,
    );

    registry
}

fn handle_get_weather(args: &Value) -> Result<String> {
    let location = args
        .get("location")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");

    // Deterministic mock data for demonstration and testing.
    let result = json!({
        "location": location,
        "temperature_celsius": 22,
        "condition": "Partly cloudy",
        "humidity_percent": 65,
    });
    Ok(result.to_string())
}

fn handle_calculate(args: &Value) -> Result<String> {
    let expr = args
        .get("expression")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if expr.is_empty() {
        bail!("Empty expression");
    }

    // Very simple evaluator: supports two operands with one operator.
    let result = evaluate_simple_expression(expr)?;
    Ok(result.to_string())
}

fn handle_read_file(args: &Value) -> Result<String> {
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");

    if path.is_empty() {
        bail!("No path provided");
    }

    let contents = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{path}': {e}"))?;
    Ok(contents)
}

/// Evaluate a simple arithmetic expression of the form `<number> <op> <number>`.
/// Supports chained operations evaluated left-to-right.
pub fn evaluate_simple_expression(expr: &str) -> Result<f64> {
    let tokens = tokenize(expr)?;
    if tokens.is_empty() {
        bail!("Empty expression");
    }

    let mut result = parse_number(&tokens[0])?;
    let mut i = 1;
    while i + 1 < tokens.len() {
        let op = tokens[i].as_str();
        let rhs = parse_number(&tokens[i + 1])?;
        result = apply_op(result, op, rhs)?;
        i += 2;
    }
    Ok(result)
}

fn tokenize(expr: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in expr.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else if "+-*/".contains(ch) && !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    Ok(tokens)
}

fn parse_number(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .map_err(|_| anyhow::anyhow!("Invalid number: {s}"))
}

fn apply_op(lhs: f64, op: &str, rhs: f64) -> Result<f64> {
    match op {
        "+" => Ok(lhs + rhs),
        "-" => Ok(lhs - rhs),
        "*" => Ok(lhs * rhs),
        "/" => {
            if rhs == 0.0 {
                bail!("Division by zero");
            }
            Ok(lhs / rhs)
        }
        _ => bail!("Unsupported operator: {op}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_simple_addition() {
        let result = evaluate_simple_expression("2 + 3").unwrap();
        assert!((result - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_multiplication() {
        let result = evaluate_simple_expression("4 * 5").unwrap();
        assert!((result - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_division() {
        let result = evaluate_simple_expression("10 / 4").unwrap();
        assert!((result - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_chained() {
        // Left-to-right: (2 + 3) * 4 = 20
        let result = evaluate_simple_expression("2 + 3 * 4").unwrap();
        // Our evaluator is left-to-right, so 2+3=5, 5*4=20
        assert!((result - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_division_by_zero() {
        let result = evaluate_simple_expression("5 / 0");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_expression() {
        let result = evaluate_simple_expression("");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_weather_handler() {
        let args = json!({"location": "Tokyo"});
        let result = handle_get_weather(&args).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["location"], "Tokyo");
        assert_eq!(parsed["temperature_celsius"], 22);
    }

    #[test]
    fn test_calculate_handler() {
        let args = json!({"expression": "10 + 5"});
        let result = handle_calculate(&args).unwrap();
        assert_eq!(result, "15");
    }

    #[test]
    fn test_read_file_empty_path() {
        let args = json!({"path": ""});
        let result = handle_read_file(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_file_nonexistent() {
        let args = json!({"path": "/tmp/nonexistent_file_12345.txt"});
        let result = handle_read_file(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_registry_execute() {
        let registry = default_tool_registry();
        let result = registry
            .execute("get_weather", r#"{"location":"Berlin"}"#)
            .unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["location"], "Berlin");
    }

    #[test]
    fn test_tool_registry_unknown_tool() {
        let registry = default_tool_registry();
        let result = registry.execute("nonexistent", "{}");
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_registry_definitions_count() {
        let registry = default_tool_registry();
        assert_eq!(registry.definitions().len(), 3);
    }
}
