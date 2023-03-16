"""
In this tutorial I am going to implement vision transformer, and here are the steps:
Step1:Patchifying and the linear mapping
we need to convert eah image into set of vectors each vector represent one of the image patches  
"""
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor


#step1: the idea is to reshape each n,c,h,w (batch size,channel,height,width)
#		into n, n_patches*n_patches , (h*w*c) / (n_patches*n_patches). then we flatten each
#		n_patch into a vector and apply a linear transformation to any embedding length 
# 		we want (512) for example, to generate tokens. Finally, we are going to concatinate
# 		the v-class token (classification token) to the previous tokens.
#step2: second step is to app the positional encoding. The role of this vector 
#		is to help the model to understand the position of each patch in the original image.
#		those embeddings can be leared, but we can instead use sines and cosines waves. 
#		In fact that is possible because, what positional vectors do is they add low frequency 
#		low frequency values to the tokens at the beginning of the image and high frequency values
#		for the tokens referring to the end of the image.(check Vaswani et. al. Attention is all
# 		you need 2017).
#step3: In this step we are going to build the most important part of the model, i.e. the Encoder Block
#		the encoder block is composed of a normalization step the tokens of each image from the image batch
#		is normalized to have 0 mean and 1 std. the second step is to add the Multi-head Attention, and finally
#		have some residual connctions to be added. those residua connections will help the system to extend without
#		having the problem of vanishing gradient. thus for this step we are going to add a new class for the 
#		self attention. we build to classes one is a MultiHeadSelfAttention with loops and another implementation 
# 		without loops. also we implement the Transformer Encoder block that is based on MHSA blocks
#step4: This is the final step where we ill add the Multi Layer Perceptron responsible for classification
#		this is done by extracting the first classification token (v_class) from the tokens list we have
#		and then apply a customized MLP based on the number of classes we have.

def patchify(images,n_patches):
	n,c,h,w =images.shape
	assert h==w, "this method only works on square images"
	patches=torch.zeros(n,n_patches * n_patches , (h*w*c)//(n_patches*n_patches))
	patch_length= h // n_patches
	for imgID,img in enumerate(images):
		for i in range(n_patches):
			for j in range(n_patches):
				patches[imgID,i*n_patches+j]= img[:, i * patch_length: (i + 1) * patch_length, j * patch_length: (j + 1) * patch_length].flatten()
	#patches=[image[:, i * patch_length: (i + 1) * patch_length, j * patch_length: (j + 1) * patch_length].flatten() for image in images for i in range(n_patches) for j in range(n_patches) ]
	#patches=torch.Tensor(patches)
	return patches
      




class MultiHeadSelfAttention_withLoop(nn.Module):
	def __init__(self,token_dim, nbr_heads ) -> None:
		super().__init__()
		self.token_dim=token_dim
		self.nbr_heads=nbr_heads
		assert token_dim % nbr_heads == 0 , f"The dimension of the tokens {token_dim} is not divisible by the nbr of heads {nbr_heads}"
		self.head_dim=token_dim // nbr_heads

		#here we are going to create the query (q), keys (k) and values (v) tensors used by the self attention
		#and since we are working with multi head attention we will have multi q,k and v. 
		# for each patch token it will be divided on the heads, and linearly transformed 
		# using the q,k and v of that head. the final results are concatinated to get an 
		# output with the same dimentions as the input. ex: suppose we have a sequence of
		# size (N,30,10), that is the size of each patch is 10 and we have 30 of them and
		# a batch size of N. Suppose that we have two heads, then each patch of size 10 is 
		# passed into the two heads, thus we will have two vectors each of size 5. at this 
		#level we will have a tensor of(N,30,2,5) dimention. finally we will concatinate this 
		# tensor on the third axis (2) and we will get back (N,30,10) output which is similar
		#to the input shape.
		self.queries= nn.ModuleList([nn.Linear(self.head_dim,self.head_dim) for _ in range(self.nbr_heads)])
		self.values= nn.ModuleList([nn.Linear(self.head_dim,self.head_dim) for _ in range(self.nbr_heads)])
		self.keys= nn.ModuleList([nn.Linear(self.head_dim,self.head_dim) for _ in range(self.nbr_heads)])
		#softmax on dim -1 means that we are going to normalize the tensor on axis -1. i.e. to say that the 
		#mean and the std are 0 and 1 for all the tensors 
		self.softmax=nn.Softmax(dim=-1)

	def forward(self,tokens):
		# here tokens represent the tokens of a batch of images
		# the input tokens are of shape (N,n_patches**2,token_dim)
		# it will be transfered into shape (N,n_patches**2,nbr_heads,token_dim//nbr_heads) 
		# after passing through the multi head attention 
		# finally and after the concatination it will return back to the initial shape (N,n_patches**2,token_dim)

		imgs_batch_output=[]
		for one_img_token_list in tokens:
			one_img_output=[]
			for head in range(self.nbr_heads):
				query=self.queries[head]
				value=self.values[head]
				key=self.keys[head]

				tokens_part=one_img_token_list[:,head * self.head_dim : (head + 1)* self.head_dim]
				q,k,v =query(tokens_part), key(tokens_part),value(tokens_part)

				attention=self.softmax(q @ k.T / (self.head_dim ** 0.5))
				one_img_output.append(attention @ v)
			imgs_batch_output.append(torch.hstack(one_img_output))
		#here the unsqueeze is to add an axis at the beginning of the tensor
		return torch.cat([torch.unsqueeze(imgs_output,dim=0) for imgs_output in imgs_batch_output])


# implementing the ViT using loops is not the best way we can do the same without loops:
class MultiHeadSelfAttention_withoutLoop(nn.Module):
	def __init__(self,token_dim, nbr_heads ) -> None:
		super().__init__()
		self.token_dim=token_dim
		self.nbr_heads=nbr_heads
		assert token_dim % nbr_heads == 0 , f"The dimension of the tokens {token_dim} is not divisible by the nbr of heads {nbr_heads}"
		self.head_dim=token_dim // nbr_heads
		# here we are creating one Linear layer with output 3 times the input dimensions
		# and this is because this Linear layer will do the work of query(q), key(k), and value(v)
		# that is to say, instead of making three different Linear layers one for each of the
		# q, k and v, we will train one Linear layer with the same number of dimensions and then we can
		# reshape the output to get the same shape that we expected when thaving three separated layers


		self.qkv=nn.Linear(self.head_dim,self.head_dim*3)
		
	def forward(self, tokens):
		batch_size,nbr_tokens,token_dim=tokens.shape
		#here ware doing the reshaping of the Linear layer to be able to get the q,k and v
		#we are also doing a permutation to rearrange the axis so that we have the value 3 in
		#the first axis and then those three will be assigned to the q,k and v
		qkv=self.qkv(tokens).reshape(batch_size,nbr_tokens,3,self.nbr_heads,token_dim // self.nbr_heads).permute(2,0,3,1,4)
		q,k,v=qkv[0],qkv[1],qkv[2]
		
		attention=(q @ k.transpose(-2,-1)) / (self.head_dim ** 0.5)
		attention=attention.softmax(dim=-1)
		tokens= (attention @ v).transpose(1,2).reshape(batch_size,nbr_tokens,token_dim)
		return tokens




# this class is the transformer encoder. it containes a normalization layer that normalizes
# the input tokens, followed by a multihead self attention block, then again normalizing the 
# output of the multihead self attention and finally a Multi Layer Perceptron (MLP). usually 
# the MLP is of two layers, the hidden layer is four times as big as the token dimension
class TransformerEncoderBlock(nn.Module):
	def __init__(self,token_dim,nbr_heads,mlp_ratio=4) -> None:
		super().__init__()
		self.token_dim = token_dim
		self.nbr_heads=nbr_heads

		self.norm1=nn.LayerNorm(token_dim)
		self.MHSA=MultiHeadSelfAttention_withLoop(token_dim,nbr_heads)
		self.norm2= nn.LayerNorm(token_dim)

		self.MLP= nn.Sequential(

			nn.Linear(token_dim,token_dim * mlp_ratio),
			nn.GELU(),
			nn.Linear(token_dim * mlp_ratio, token_dim)
		)

	def forward(self,tokens):
		#here we are doing the residual, by adding the tokens to output of the MHSA
		#and also another residual by adding the results to the outpu of the MLP 
		result=tokens +self.MHSA(self.norm1(tokens)) 
		result= result + self.MLP(self.norm2(result))
		return result



class VisionTransformer(nn.Module):
	#output dimention is the final output of the model and it should be equal to the number of classes
	def __init__(self,chw=(1,640,640),n_patches=20,token_dim=512,nbr_blocks=4,nbr_heads=10,output_dim=10):
		super(VisionTransformer,self).__init__()
		self.chw=chw
		self.n_paches=n_patches
		self.nbr_blocks=nbr_blocks
		self.nbr_heads=nbr_heads
		self.token_dim=token_dim


		assert chw[1] % n_patches == 0 and chw[2] % n_patches ==0, "the number of paches should be compatile with the size of the image"   
		self.patch_size=(chw[1]//n_patches, chw[2]//n_patches)

		# 1) here we are going to apply a linear transformation on the patches to have an embedding with a given length for each patch
		self.Flatten_input_dim=int(chw[0]*self.patch_size[0]*self.patch_size[1])
		self.Linear_layer_1=nn.Linear(self.Flatten_input_dim,self.token_dim)

		# 2) after that we are going to add a learnable token that will be concatinated
		#to the other tokens this token will embed inside all the required information
		#that are necessary to apply the classification (supposing we have a classification problem)
		# and we can add as much as we want of this learnable token based on the number of tasks
		# we have (for example if we want to classify the MNIST images into their value (0-9) and 
		# at the same time we want to have an output saying if the number is greater or equal to five 
		# we can have two learnable tokens added in this case each one will learn the necessary information
		#to do one of the classifications)
		self.v_class=nn.Parameter(torch.rand(1,self.token_dim))


		# 3) we are going to create the positional embeddings and make them non trainable
		#self.positional_embedding= nn.Parameter(torch.tensor(self.generate_positional_embeddings()))
		#self.positional_embedding.requires_grad= False
		self.register_buffer('positional_embeddings', self.generate_positional_embeddings(), persistent=False)
		# 4) Transformer Encoder Bolcks
		self.TEBs=nn.ModuleList([TransformerEncoderBlock(token_dim,nbr_heads) for _ in range(nbr_blocks)])

		# 5) Here we are going to add the classification MPL
		self.Classification_MLP=nn.Sequential(
			nn.Linear(self.token_dim,output_dim),
			nn.Softmax(dim=-1)
		)

	def forward(self, images):
		n,c,h,w=images.shape
		patches=patchify(images,self.n_paches).to(self.positional_embeddings.device)
		tokens=self.Linear_layer_1(patches)

		#the v_class is concatinated with the other tokens, and it is concatinated at the beginning 
		#of the tokens. we are going to use this when we want to send its values to the FCNN to do the 
		#classification. we can do it by loop or without loop as shown bellow
		#tokens=torch.vstack([(self.v_class,tokens[i]) for i in range(len(tokens))])
		tokens=torch.cat((self.v_class.expand(n,1,-1),tokens),dim=1)
		
		#we are going to add the positional tokens to the image patch tokens 
		#but we should pay attention to the batch size, i.e. the size of the token is (N,n_patchess**2 +1,token_dim)
		# where N is the batch size. thus we need to repeat the positional imbeddings for N times.
		positional_embeddings=self.positional_embeddings.repeat(n,1,1)
		results=tokens+positional_embeddings


		for block in self.TEBs:
			results=block(results)

		v_class= results[:, 0 , :]
		return self.Classification_MLP(v_class)
	

	def generate_positional_embeddings(self):
		nb_tokens=self.n_paches**2+1
		positions_embedding=torch.ones(nb_tokens,self.token_dim)
		for i in range(nb_tokens):
			for j in range(self.token_dim):
				if j % 2 ==0:
					positions_embedding[i][j]=np.sin(i / (10000 ** (j / nb_tokens )))
				else:
					positions_embedding[i][j]=np.cos(i / (10000 ** ( (j - 1) / nb_tokens )))
		return positions_embedding
	


if __name__ == '__main__':
	model = VisionTransformer()
	x=torch.randn(7,1,640,640)
	print(model(x).shape)