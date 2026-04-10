use anyhow::{Error, Result};
use serde_json::{Value, json};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut history: Vec<Value> = vec![];
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut reader = BufReader::new(stdin);

    loop {
        stdout.write_all(b"> ").await?;
        stdout.flush().await?;
        let mut input = String::new();
        reader.read_line(&mut input).await?;
        history.push(json!({"role": "user", "content": input.trim_end().to_string()}));
        agent_loop(&mut history).await;
        if let Some(last) = history.last() {
            if let Some(Value::Array(arr)) = last.get("content") {
                for block in arr {
                    if let Some(text) = block.get("text") {
                        if let Some(str) = text.as_str() {
                            stdout.write_all(str.as_bytes()).await?;
                            stdout.flush().await?;
                        }
                    }
                }
            }
        }
    }
}

async fn agent_loop(message: &mut Vec<Value>) {}
