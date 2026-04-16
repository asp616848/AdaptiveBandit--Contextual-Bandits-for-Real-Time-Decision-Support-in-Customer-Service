cd /data/interns/studentiotlab/AdaptiveBandit
python - <<"PY"
import json
from pathlib import Path
root = Path('Simulation_4/artifacts/phase10_prod_linux2')
report = json.loads((root / 'full_pipeline_report.json').read_text())
val = report.get('validation', {})
ppo = report['evaluation']['ppo']
doc = report['evaluation']['baselines']['document_guided']
print('validation_passed_checks', val.get('passed_checks'))
print('validation_total_checks', val.get('total_checks'))
print('validation_overall_pass', val.get('overall_pass'))
print('ppo_mean_reward', round(float(ppo['mean_reward']), 6))
print('ppo_resolution_rate', round(float(ppo['resolution_rate']), 6))
print('ppo_escalation_rate', round(float(ppo['escalation_rate']), 6))
print('ppo_dropout_rate', round(float(ppo['dropout_rate']), 6))
print('doc_mean_reward', round(float(doc['mean_reward']), 6))
print('doc_resolution_rate', round(float(doc['resolution_rate']), 6))
print('ppo_beats_document_guided', report['evaluation']['ppo_beats_document_guided'])
train_summary = json.loads((root / 'training_summary.json').read_text())
print('best_eval_reward', round(float(train_summary.get('best_eval_reward', 0.0)), 6))
print('final_resolution_rate_train_summary', round(float(train_summary.get('final_resolution_rate', 0.0)), 6))
print('final_num_timesteps', train_summary.get('final_num_timesteps'))
PY