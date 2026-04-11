use anyhow::{Error, Result};
use reqwest::Client;
use dotenvy::dotenv;
use serde_json::{Value, json};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

struct Config {
    model: String,
    url: String,
    api_key: String,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenv()?;
    let config = Config {
        model: std::env::var("MODEL")?,
        url: std::env::var("URL")?,
        api_key: std::env::var("API_KEY")?,
    };
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
        let _ = agent_loop(&mut history, &config).await;
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

async fn agent_loop(message: &mut Vec<Value>, config: &Config) -> Result<(), Error> {
    let client = Client::new();
    let response = client
        .post(&config.url)
        .header(
            "Authorization",
            format!(
                "Bearer {}",
                config.api_key
            ),
        )
        .json(
            &json!({"model": config.model, "messages": message})
        )
        .send()
        .await?;
    let json = response.json::<Value>().await?;
    dbg!(&json);
    Ok(())
}
