#[derive(Clone, Debug)]
pub(crate) struct Health;

impl Health {
    pub(crate) fn new() -> Self {
        Self
    }
    pub(crate) async fn check(&mut self) -> bool {
        true
    }
}
