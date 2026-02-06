# Hyperparameters
LR = 1e-3
EPOCHS = 20
LATENT_SIZE = 16
CONTEXT_SIZE = 64
NUMBER_OF_TARGETS_FOR_PREDICTION = 4
WINDOW_SIZE = 50
STRIDE = 10
HORIZONS = [4,8,16]
SEQUENCE_LENGTH = 10
VALIDATION_RATIO = 0.2
BATCH_SIZE = 32
PREDICTION = "hrv difference"


# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="eml-labs",
    # Set the wandb project where this run will be logged.
    project="Prediction-PAF-Onset-using-CNN-sliding-window-using-RR-intervals",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": LR,
        "epochs": EPOCHS,
        "latent_size": LATENT_SIZE,
        "context_size": CONTEXT_SIZE,
        "number_of_targets_for_prediction": NUMBER_OF_TARGETS_FOR_PREDICTION,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "horizons": HORIZONS,
        "sequence_length": SEQUENCE_LENGTH,
        "validation_ratio": VALIDATION_RATIO,
        "batch_size": BATCH_SIZE,
        "prediction": PREDICTION,
    },
)

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    batch_count = 0

    for rr_windows, hrv_targets in train_loader:
        rr_windows = rr_windows.to(device)
        hrv_targets = hrv_targets.to(device)
        optimizer.zero_grad()
        loss,loss_1,loss_2,loss_4 = training_step(model, rr_windows, hrv_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_count += 1

    train_loss = running_loss / batch_count
    val_loss,val_loss_1,val_loss_2,val_loss_4 = validation_step(model, val_loader, device)
    run.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_loss_1": loss_1.item(),
        "train_loss_2": loss_2.item(),
        "train_loss_4": loss_4.item(),
        "val_loss": val_loss,
        "val_loss_1": val_loss_1,
        "val_loss_2": val_loss_2,
        "val_loss_4": val_loss_4
    })
    print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.6f} Validation Loss = {val_loss:.6f} 1 : {val_loss_1:.6f} 2 : {val_loss_2:.6f} 4 : {val_loss_4:.6f}")

