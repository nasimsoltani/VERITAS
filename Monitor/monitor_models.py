import torch.nn as nn
import torch

class MonitorNet(nn.Module):
	""" input shape needs to be (b, 2, 90, 36) """

	def __init__(self):
		super(MonitorNet, self).__init__()
		dropProb = 0.25
		channel = 32 
		#self.channel_param = channel_param

		self.conv1 = nn.Conv2d(2, channel, kernel_size=(28,28), padding="same")
		self.conv2 = nn.Conv2d(channel, channel, kernel_size=(7, 7), padding="same")
		self.conv3 = nn.Conv2d(channel, channel, kernel_size=(7, 7), padding="same")
		self.conv4 = nn.Conv2d(channel, channel, kernel_size=(7, 7), padding="same")
		self.conv5 = nn.Conv2d(channel, channel, kernel_size=(7, 7), padding="same")
		self.conv6 = nn.Conv2d(channel, channel, kernel_size=(7, 7), padding="same")
		self.conv7 = nn.Conv2d(channel, channel, kernel_size=(7, 7), padding="same")
		self.pool1 = nn.MaxPool2d((2,2))
		#self.pool2 = nn.MaxPool2d((3, 3), padding=1)
		self.hidden1 = nn.Linear(1408, 256) 
		"""if self.channel_param == 'Delay_Spread':
			#self.hidden1 = nn.Linear(1408, 512) 
			self.hidden2 = nn.Linear(256, 256)  """
		
		self.drop = nn.Dropout(dropProb)
		self.relu = nn.ReLU() 		#nn.LeakyReLU(0.1)    #nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv1(x))
		
		a = x
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))     
		x = torch.add(x, a)
		x = self.pool1(x)
		#x = self.drop(x)

		b = x
		x = self.relu(self.conv4(x))
		x = self.relu(self.conv5(x))     
		x = torch.add(x, b)
		x = self.pool1(x)
		#x = self.drop(x)

		c = x 
		x = self.relu(self.conv6(x))
		x = self.relu(self.conv7(x))
		x = torch.add(x, c)
		x = self.pool1(x)
		
		x = self.drop(x)

		x = x.view(x.size(0), -1)
		x = self.hidden1(x)

		"""if self.channel_param == 'Delay_Spread':
			x = self.relu(x)
			x = self.hidden2(x)"""

		x = x/(torch.max(torch.abs(x)))

		return x


if __name__ == '__main__':

	channel_param = 'Delay_Spread'
	# create the model:
	model = MonitorNet()
	# print number of parameters in the model
	pp=0
	for p in list(model.parameters()):
		n=1
		for s in list(p.size()):
			n = n*s
		pp += n
	print('This model has ' +str(pp)+ ' parameters')

	"""weight_path = '/home/ns38942/AiR/results/weights-OOD_AM_BOCRBPSK_CW_LFM_PCW.pt' 
	state_dict = torch.load(weight_path)
	model.load_state_dict(state_dict)"""
	
	
	dummy_input = torch.rand((1,2,90,36))  
	dummy_output = model(dummy_input)

	print(dummy_output.shape)


