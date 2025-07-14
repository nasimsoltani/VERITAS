import torch.nn as nn
import torch
import os
import pickle

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:

        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


def train_model(model, train_dl, val_dl, channel_param, save_path):
	
	epochs = 200           # Number of epochs you want to train for
	early_stopping = True  # Set to True or False to enable or disable early stopping
	# If early_stopping is enabled, patience shows the number of consecutive epochs
	# after which training stops if training loss does not improve.
	patience = 8 
	lr_patience = 4

	# do the model/optimizer/loss function initialization

	# import timm
	# from pytorch_metric_learning import losses

	learning_rate = 1e-4
	min_learning_rate = 1e-7 
	
	if channel_param == 'Channel_Profile':
		min_learning_rate = 1e-4
	"""elif channel_param == 'Tx_Speed':
		min_learning_rate = 1e-7 
	elif channel_param == 'Delay_Spread':
		min_learning_rate = 1e-7 """

	model.train()
	model.cuda()

	# loss_func = torch.nn.CrossEntropyLoss()
	loss_func = torch.jit.script(TripletLoss())
	# loss_func = losses.ContrastiveLoss()
	# loss_func = torch.jit.script(contrastive())

	# loss_func = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

	model_path = os.path.join(save_path,'weights_' + channel_param +'.pt')

	# Now do conventional training
	# train the model
	best_vloss = 1000000
	epoch_patience_cntr = 0 
	previous_epoch_vloss = 100000

	train_loss_list, val_loss_list = [], []
	train_acc_list, val_acc_list = [], []

	for epoch in range(epochs):  # First for loop: loop over the dataset multiple times (epochs)
		print('epoch ' + str(epoch+1) + '/' + str(epochs))

		running_loss = 0.0 
		last_loss = 0.0 
		all_batches_acc = 0 
		all_batches_loss = 0 

		for batch_cntr, (anchor_input, positive_input, negative_input, labels) in enumerate(train_dl):   # Second for loop: in each epoch, loop over batches

			# Every data instance is an input + label pair
			#         anchor_input, positive_input, negative_input, labels = data
			anchor_input = anchor_input.float().cuda()
			positive_input = positive_input.float().cuda()
			negative_input = negative_input.float().cuda()
			labels = labels.long().cuda()
			
			

			# pass training batch through the model
			anchor_outputs = model(anchor_input)
			positive_outputs = model(positive_input)
			negative_outputs = model(negative_input)


			# Zero your gradients for every batch
			optimizer.zero_grad()
			# compute loss and do the backward path
			# triplet loss
			loss = loss_func(anchor_outputs, positive_outputs, negative_outputs)
			# loss = loss_func(anchor_outputs, negative_outputs, labels)
			loss.backward()
			# Adjust learning weights
			optimizer.step()

			# now that one round of backward is done, test the model on this training batch to calculated training loss and accuracy
			# passing the batch through the model
			anchor_outputs = model(anchor_input)    
			positive_outputs = model(positive_input)
			negative_outputs = model(negative_input)
			# triplet loss:
			loss = loss_func(anchor_outputs, positive_outputs, negative_outputs)
			#         loss = loss_func(anchor_outputs, negative_outputs, labels)
			# compute batch accuracy, and add it to all_batches_acc:
			all_batches_loss += loss.item()

			# Gather loss for reporting just now (running)
			running_loss += loss.item()


			if batch_cntr % 1000 == 999:   # print loss every 1000 batches to avoid over-crowding the logs
				last_loss = running_loss / 1000 # loss per batch
				print('  batch {} loss: {}'.format(batch_cntr + 1, last_loss))
				running_loss = 0.

		# training for one epoch is done over all batches
		# Now calculate training set loss and accuracy
		avg_loss = all_batches_loss/(batch_cntr+1)
		avg_acc = all_batches_acc/(batch_cntr+1)

		# training for one epoch (all batches) is done, now do the validation
		# put the model in evaluation (test) mode
		model.eval()

		with torch.no_grad():     # We don't need gradients on to do reporting

			running_vloss = 0.0 
			all_batches_v_acc = 0 

			for batch_cntr, vdata in enumerate(val_dl):
				
				vinputs_anchor, vinputs_positive, vinputs_negative, vlabels = vdata

				vinputs_anchor = vinputs_anchor.float().cuda()
				vinputs_positive = vinputs_positive.float().cuda()
				vinputs_negative = vinputs_negative.float().cuda()
				vlabels = vlabels.long().cuda()

				voutputs_anchor = model(vinputs_anchor)
				voutputs_positive = model(vinputs_positive)
				voutputs_negative = model(vinputs_negative)

				# triplet loss
				vloss = loss_func(voutputs_anchor, voutputs_positive, voutputs_negative)
				#             vloss = loss_func(voutputs_anchor, voutputs_negative, vlabels)
				running_vloss += vloss

			avg_vloss = running_vloss / (batch_cntr+1)
			print('********* validation Loss = '+str(avg_vloss)+' ************')
			print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
			train_loss_list.append(avg_loss)
			val_loss_list.append(avg_vloss.cpu().detach().numpy())

			# Track best performance, and save the model's state
			if avg_vloss < best_vloss:
				epoch_patience_cntr = 0
				best_vloss = avg_vloss
				torch.save(model.state_dict(), model_path)
			else:
				print('*** validation loss did not improve *** ' + str(epoch_patience_cntr))
				print('best_vloss: ' +str(best_vloss))
				epoch_patience_cntr += 1

			torch.cuda.empty_cache()

			# save the model at the end of the epoch any way
			# torch.save(model.state_dict(), model_path)

			if epoch_patience_cntr == lr_patience and learning_rate >= min_learning_rate*2:    # multiply the learning rate by 0.5 if validation loss does not improve for 3 epochs
				print(' -------------- reducing learning rate')
				learning_rate = learning_rate * 0.5
				print('new learning_rate = '+str(learning_rate))
				# reconfigure the optimizer:
				epoch_patience_cntr = 0
				optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)


			if early_stopping and epoch_patience_cntr == patience + 1:
				# stop training
				print('********** stopping training **********')
				torch.save(model.state_dict(), model_path)
				break

	# Conventional training for all epochs has ended

	print(train_loss_list)
	print('-------------------')
	val_loss_list = list(map(lambda x: float(x), val_loss_list))
	print(val_loss_list)

	loss_dict = {'train_loss':train_loss_list, 'val_loss':val_loss_list}

	with open('loss_' + channel_param +'.pkl','wb') as handle:
		pickle.dump(loss_dict,handle)


