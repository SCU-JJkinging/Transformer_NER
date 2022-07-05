# import sys
# sys.path.append('/home/seatrend/jinxiang/Efficient_GlobalPointer')
import os
import torch
from run_model.globalpointer_util import evaluate, train
from train_config.globalpointer_config import Config
from transformers import AdamW, get_linear_schedule_with_warmup
from data_processing.data_process import yeild_data
from model.model import EfficientGlobalPointerNet as GlobalPointerNet
from loss_function.loss_fun import global_pointer_crossentropy
from utils.tools import EMA
from utils.tools import setup_seed


def main():
    setup_seed(1234)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))

    config = Config()

    label_list = ['pro', 'dis', 'sym', 'ite', 'bod', 'dru', 'mic', 'equ', 'dep']
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_dataloader = yeild_data(config.train_file_data, label2id, model_type='global_pointer', is_train=True,
                                  ddp=False)
    val_dataloader = yeild_data(config.val_file_data, label2id, model_type='global_pointer', is_train=False,
                                ddp=False)

    model = GlobalPointerNet(config.pretrained_model_path, len(label_list), config.head_size,
                             config.hidden_size, config, device).to(device)

    # ema = EMA(model, 0.999)
    # ema.register()

    total_steps = len(train_dataloader) * config.epochs // config.accumulate_step
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_prop * total_steps,
                                                num_training_steps=total_steps)

    best_score = 0
    start_epoch = 1
    # Data for loss curves plot.
    epochs_list = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument.
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_list = checkpoint["epochs_list"]
        train_losses = checkpoint["train_loss"]
        valid_losses = checkpoint["valid_loss"]

    # Compute loss and accuracy before starting (or resuming) training.
    # val_time, val_loss, val_f1 = evaluate(val_dataloader, global_pointer_crossentropy, model)
    #
    # print("-> Valid time: {:.4f}s loss = {:.6f} val_f1: {:.6f}".format(val_time, val_loss, val_f1))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training Model model on device: {}".format(device),
          20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, config.epochs + 1):
        epochs_list.append(epoch)
        print("* Training epoch {}:".format(epoch))
        train_time, train_loss, train_f1 = train(train_dataloader, model, global_pointer_crossentropy, optimizer,
                                                 scheduler)
        train_losses.append(train_loss)
        print(
            "-> Training time: {:.4f}s train_loss = {:.6f} train_f1 = {:.6f}".format(train_time, train_loss, train_f1))
        with open('../result/metric.txt', 'a', encoding='utf-8') as fp:
            fp.write(
                'Epoch:' + str(epoch) + '\t' + 'train_loss:' + str(round(train_loss, 6)) + '\t' + 'train_f1:' + str(
                    round(train_f1.item(), 6)) + '\t')

        val_time, val_loss, val_f1 = evaluate(val_dataloader, global_pointer_crossentropy, model)
        print("-> Valid time: {:.4f}s val_loss = {:.6f} val_f1: {:.6f}".format(val_time, val_loss, val_f1))

        with open('../result/metric.txt', 'a', encoding='utf-8') as fp:
            fp.write('val_loss:' + str(round(val_loss, 6)) + '\t' + 'val_f1:' +
                     str(round(val_f1.item(), 6)) + '\n')

        valid_losses.append(valid_losses)

        # Early stopping on validation accuracy.
        if val_f1 < best_score:
            patience_counter += 1
        else:
            best_score = val_f1
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_list": epochs_list,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(config.target_dir, "model_best.pth.tar"))

        # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_list": epochs_list,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(config.target_dir, "model_{}.pth.tar".format(epoch)))

        if patience_counter >= config.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == '__main__':
    main()
