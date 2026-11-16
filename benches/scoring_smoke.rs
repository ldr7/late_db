use criterion::{criterion_group, criterion_main, Criterion};
use late_db::scoring::TodoScorer;
use late_db::{EngineConfig, LateInteractionEngine};

fn scoring_smoke(c: &mut Criterion) {
    c.bench_function("scoring_scaffold_smoke", |b| {
        b.iter(|| {
            let config = EngineConfig::local_default(128);
            let _engine = LateInteractionEngine::new(config, TodoScorer).expect("engine scaffold");
        });
    });
}

criterion_group!(benches, scoring_smoke);
criterion_main!(benches);
