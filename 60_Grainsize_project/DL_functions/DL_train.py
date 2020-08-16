# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:29:38 2020

@author: hto_r
"""

import torch


def train (train_generator, test_generator, criterion, model, epochs, optimizer, Batch_size):
    """Function to train a pytorch model
    Args:
        train_generator: pytorch train generator instance
        test_generator: pytorch test generator instance
        criterion: pytorch criterion
        model: pytorch model
        epochs: int, number of epochs to train
        optmizer: pytorch optmizer
        Batch_size: int, batch size for forward passes
    
    Returns:
        train_losses: list with the train losses 
        validation_losses: list with the validation losses
        accuracy_vect: list with the accuracy of the classification of the test
        
    """
    
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses, validation_losses =[], []
    accuracy_vect=[]
    
    for e in range (epochs):
        print('epoch ', e)
        running_loss=0
        
        for i in range(1):
            
            images, labels =next(iter(train_generator)) 
            images, labels= images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            log_ps=model(images)
            loss=criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss+= loss.item()
        
        else:
            
            accuracy=0
            validation_loss=0
            
            with torch.no_grad():
                model.eval()
                for i in range(1):
                    images, labels =next(iter(test_generator)) 
                    images, labels= images.to(device), labels.to(device)
                    log_ps=model(images)
                    validation_loss+=criterion(log_ps, labels).item()
                    ps=torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    print(torch.reshape(top_class, (1, -1)))
                    print(labels)
                    hits=top_class==labels.view(*top_class.shape)
                    accuracy+=torch.mean(hits.type(torch.FloatTensor))
                    
            model.train()
            train_losses.append(running_loss/Batch_size)
            validation_losses.append(validation_loss/Batch_size)
            accuracy_vect.append(accuracy.numpy().item(0))
        
    return(train_losses, validation_losses, accuracy_vect)