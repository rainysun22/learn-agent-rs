use anyhow::Result;
use dotenvy::dotenv;
use learn_agent_rs::agent::{Agent, AgentConfig};
use learn_agent_rs::tools::default_tool_registry;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok(); // .env is optional; env vars may be set directly

    let config = AgentConfig {
        model: std::env::var("MODEL")?,
        url: std::env::var("URL")?,
        api_key: std::env::var("API_KEY")?,
        max_iterations: std::env::var("MAX_ITERATIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10),
    };

    let system_prompt = "You are a helpful assistant. You have access to tools you can use \
                         to help answer questions. When you need external information or \
                         computation, use the available tools.";

    let mut agent = Agent::new(config, Some(system_prompt));
    let tools = default_tool_registry();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut reader = BufReader::new(stdin);

    loop {
        stdout.write_all(b"\n> ").await?;
        stdout.flush().await?;

        let mut input = String::new();
        let bytes_read = reader.read_line(&mut input).await?;

        // EOF (Ctrl-D) — exit gracefully
        if bytes_read == 0 {
            stdout.write_all(b"\nGoodbye!\n").await?;
            stdout.flush().await?;
            break;
        }

        let trimmed = input.trim();

        // Skip empty lines
        if trimmed.is_empty() {
            continue;
        }

        // Allow explicit exit commands
        if matches!(trimmed, "exit" | "quit" | "q") {
            stdout.write_all(b"Goodbye!\n").await?;
            stdout.flush().await?;
            break;
        }

        match agent.chat(trimmed, &tools).await {
            Ok(Some(response)) => {
                stdout.write_all(response.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
            Ok(None) => {
                stdout
                    .write_all(b"(assistant produced no text response)\n")
                    .await?;
                stdout.flush().await?;
            }
            Err(e) => {
                let msg = format!("Error: {e:#}\n");
                stdout.write_all(msg.as_bytes()).await?;
                stdout.flush().await?;
            }
        }
    }

    Ok(())
}
