use crate::message::Message;

#[derive(Debug)]
pub struct Context {
    prompt: String,
    messages: Vec<Message>,
}

impl Context {
    pub fn to_messages(&self) -> Vec<Message> {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        messages.push(Message::system(self.prompt.clone()));
        messages.extend_from_slice(&self.messages);
        messages
    }

    pub fn as_messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn add(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    pub fn new(prompt: String, messages: Vec<Message>) -> Self {
        Self { prompt, messages }
    }
}
