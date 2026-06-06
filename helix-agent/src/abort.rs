pub struct AbortSignal {
    inner: tokio::sync::watch::Receiver<bool>,
}

impl AbortSignal {
    pub async fn wait(&self) {
        let _ = self.inner.clone().wait_for(|x| *x).await;
    }
}
