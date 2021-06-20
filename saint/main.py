import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Preprocess, SAINTDataset, collate_fn
from saint import SAINTModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_batch(batch):
    assessmentItemID, testId, KnowledgeTag, mask, labels = batch
    mask = ~mask
    mask = mask.type(torch.FloatTensor)
    labels = labels.type(torch.FloatTensor)
    
    interaction = labels + 1
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)
    
    assessmentItemID = ((assessmentItemID + 1) * mask).to(torch.int64)
    testId = ((testId + 1) * mask).to(torch.int64)
    KnowledgeTag = ((KnowledgeTag + 1) * mask).to(torch.int64)
    
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1
    
    mask = mask.type(torch.bool)
    mask = ~mask
    
    return (assessmentItemID, testId, KnowledgeTag, mask, labels, interaction, gather_index)

def train(train_loader, valid_loader, args):
    model = SAINTModel()
    model.to(device)
    criterion = nn.BCELoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    n_epoch = args.num_epochs

    for epoch in tqdm(range(n_epoch)):
        model.train()
        batch_loss = []
        for batch in train_loader:
            # assessmentItemID, testId, KnowledgeTag, mask, labels = batch
            assessmentItemID, testId, KnowledgeTag, mask, labels, interaction, gather_index = process_batch(batch)
            assessmentItemID, testId, KnowledgeTag, mask, labels = assessmentItemID.to(device), testId.to(device), KnowledgeTag.to(device), mask.to(device), labels.to(device)
            interaction, gather_index = interaction.to(device), gather_index.to(device)
            
            # Forward
            y_pred = model(interaction, assessmentItemID, testId, KnowledgeTag, mask)
            y_pred = y_pred.squeeze(-1)
            loss_out = criterion(y_pred, labels.to(device))
            loss_out = torch.mean(loss_out[:, -1])
            batch_loss.append(loss_out.item())
            # Update
            loss_out.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validate
        val_loss = []
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                assessmentItemID, testId, KnowledgeTag, mask, labels, interaction, gather_index = process_batch(batch)
                assessmentItemID, testId, KnowledgeTag, mask, labels = assessmentItemID.to(device), testId.to(device), KnowledgeTag.to(device), mask.to(device), labels.to(device)
                interaction, gather_index = interaction.to(device), gather_index.to(device)
                y_pred = model(interaction, assessmentItemID, testId, KnowledgeTag, mask)
                y_pred = y_pred.squeeze(-1)
                loss_out = criterion(y_pred, labels.to(device))
                loss_out = torch.mean(loss_out[:, -1])
                val_loss.append(loss_out.item())

        # Log
        print(f"Epoch: {epoch} | Train Epoch Loss: {np.mean(batch_loss)} | Val Loss: {np.mean(val_loss)}")
            
    model.eval()
    print("Done.")
    return model

def inference(model, test_loader):
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            assessmentItemID, testId, KnowledgeTag, mask, labels, interaction, gather_index = process_batch(batch)
            assessmentItemID, testId, KnowledgeTag, mask, labels = assessmentItemID.to(device), testId.to(device), KnowledgeTag.to(device), mask.to(device), labels.to(device)
            interaction, gather_index = interaction.to(device), gather_index.to(device)
            y_pred = model(interaction, assessmentItemID, testId, KnowledgeTag, mask)
            y_pred = y_pred.squeeze(-1)
            loss_out = criterion(y_pred, labels.to(device))
            y_pred = y_pred.cpu().detach().numpy()
            y_pred_last = y_pred[:, -1]
            test_preds.append(y_pred_last)
    test_preds = np.concatenate(test_preds)
    return test_preds

if __name__ == "__main__":
    print("Running this program will train the model and make a submission file")
    # Configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./input/train.csv")
    parser.add_argument("--test_path", type=str, default="./input/test.csv")
    parser.add_argument("--submission_path", type=str, default="./input/submission.csv")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--submission_out_path", type=str, default="./submission_out.csv")
    args = parser.parse_args()

    # Load data
    df_train = pd.read_csv(args.train_path)
    df_test = pd.read_csv(args.test_path)
    df_submission = pd.read_csv(args.submission_path)
    # Process data
    preprocessor = Preprocess(df_train, df_test, max_len=args.max_len)
    train_dataset, valid_dataset, test_dataset = preprocessor.create_dataset()
    # Create torch loader and dataset
    train_dataset_ = SAINTDataset(train_dataset)
    valid_dataset_ = SAINTDataset(valid_dataset)
    test_dataset_ = SAINTDataset(test_dataset)
    train_loader = DataLoader(train_dataset_, batch_size=args.train_batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset_, batch_size=args.valid_batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset_, batch_size=args.test_batch_size, collate_fn=collate_fn)

    # Train
    model = train(train_loader, valid_loader, args)
    # Test
    test_preds = inference(model, test_loader)
    df_submission['prediction'] = test_preds
    print("Submission results")
    print(df_submission['prediction'].describe())
    df_submission.to_csv(args.submission_out_path, index=False)
    print("Submission saved.")
