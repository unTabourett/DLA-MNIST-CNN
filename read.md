wandb login
poetry run python src/train.py --learning_rate 0.001 --num_epochs 10 --batch_size 32 --experiment_name "COUCOU"

poetry run python src/evaluate.py --model_path models/cnn_model.pth --batch_size 32 --experiment_name "evaluation_experiment"


