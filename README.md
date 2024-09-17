# Backpropagation of MultiLayer Perceptron

 All messages were implemented in Linear Layer, ReLU Layer, Softmax Layer and Loss Cross Entropy in attached code. 
 The actual backpropagation is performed by the gradient method in MLP. This MLP classifier has been trained on 
 spiral dataset using Stochastic Gradient Descent. I modified the layers by defining a compound layer combining softmax
 with cross-entropy loss so they can cope with numerical instabilities
 like underflow, overflow and division of zero. The result of MLP is shown on picture where you can see convergence in training progress for three 
 settings of the learning rate α ∈ {0.2, 1, 5}<br />

 ![obrazek](https://github.com/user-attachments/assets/54e83d3c-c14d-4f8f-acdb-161aec37f215)<br />

The learning rate α = 0.2 is small and results in a small improvement in accuracy
 during training progress. The accuracy would converge to the intended optimum over
 additional epochs. It would take probably two or three times as many epochs to approach it.
 Figure 2 displays the final classification.<br />

 ![obrazek](https://github.com/user-attachments/assets/295dadc9-5cee-4299-bdf7-287019b273d5)<br />

 The learning rate α = 1.0 has better results, because the improvement in accuracy over
 the epochs is a bit larger. In 1000 epochs almost 1.0 accuracy was reached. The picture shows
 the classification, where the trained model is able to clearly separate the map of the spiral.<br />

![obrazek](https://github.com/user-attachments/assets/4dd4b9b3-94d9-4f7a-80d7-c0caaae9a3b7)<br />

The learning rate α = 5.0 has too large step which causes it to initially converge
 quickly towards the optimum. However, because the learning step is too large, the accuracy
 falls to zero, leading to divergence. The picture displays the classification that was produced.<br />

![obrazek](https://github.com/user-attachments/assets/e9689470-0718-42c2-beb2-fe8385e6c5f8)<br />

 As a result, the ideal learning step α for 1000 epochs and MLP classifier with gradient
 method is around 1.0.

 # Math 

 By implementing a compound layer combining the softmax layer and multinomial
 cross-entropy loss, we can decrease the number of operations needed to compute the updates.
 The forward message for a compound layer composed of the softmax layer and multinomial cross-entropy loss. At the beginning, we put
 softmax to the loss.<br />
![obrazek](https://github.com/user-attachments/assets/fad41173-4e2b-41ab-9952-949be2744827)<br />
![obrazek](https://github.com/user-attachments/assets/e245857c-b669-4403-bc05-bd7eb26912cf)<br />
where s = (s1, . . . , sK), k ∈ {1, . . . , K} are softmax inputs, t = (t1, . . . , tK) a
 one-hot encoded vector of targets and the value of loss for a single sample.
 The backward message for a compound layer composed of the softmax layer and
 multinomial cross-entropy loss.<br />
 ![obrazek](https://github.com/user-attachments/assets/354f1b6b-138b-46ba-972d-65db75032ff7)<br />
  where p = (p1, . . . , pK), k ∈ {1, . . . , K} are softmax outputs, t = (t1, . . . , tK) a
 one-hot encoded vector of targets and δi the sensitivity of loss for a single sample.
 Softmax is invariant to shift in input, proof : pk(s′) = pk(s) s′k = sk + c for k ∈ {1, . . .
 , K} and c ∈R.<br />
 ![obrazek](https://github.com/user-attachments/assets/ae7ded8b-2ddc-467c-af22-3cfda0c0c28b)<br />

 At the end the LossCrossEntropyForSoftmaxLogits
 class was implemented with the results from previous code. This class was used to train MLP on MNIST dataset.
 NIST is a database of handwritten digits (grayscale images of 28 × 28 pixels) containing 60, 000 training and 10, 000 test examples.
 The plot showing the development of mean weight amplitude (mean absolute value)
 normalized w.r.t. the initial mean amplitude for each linear layer over 100 training epochs can be seen on picture.<br />
![obrazek](https://github.com/user-attachments/assets/fea7a0a1-0a55-4bff-8b13-4155d2348e85)<br />

 The functions have almost logarithmic growth, where the steps rapidly reduce with an
 increasing number of epochs. So at the beginning there are large errors and gradients which
 can be seen in Figure 6. After that the accuracy improves slowly. At the end the model
 reached 98.7% on training data and 94.2% on testing data.

 ![obrazek](https://github.com/user-attachments/assets/895d265a-d37b-4c42-aab3-105efe824d3d)

  The Linear_OUT is the one that affects the inputs the most because it is the last linear
 layer before the softmax combined with multinominal cross-entropy loss. As a result, it is the
 first linear layer to be visited during backpropagation.. The inputs of the gradients of the
 other linear layers are diluted by delta vectors that are backpropagated from the layers that
 come after




