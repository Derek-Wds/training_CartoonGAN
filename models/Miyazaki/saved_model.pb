��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*1.14.02v1.14.0-rc1-22-gaf24dc91b58
�
conv1_1/kernelVarHandleOp*
shape:@*
shared_nameconv1_1/kernel*
dtype0*
_output_shapes
: 
�
"conv1_1/kernel/Read/ReadVariableOpReadVariableOpconv1_1/kernel*!
_class
loc:@conv1_1/kernel*
dtype0*&
_output_shapes
:@
p
conv1_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*
shared_nameconv1_1/bias
�
 conv1_1/bias/Read/ReadVariableOpReadVariableOpconv1_1/bias*
_class
loc:@conv1_1/bias*
dtype0*
_output_shapes
:@
n
norm1/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm1/gamma
�
norm1/gamma/Read/ReadVariableOpReadVariableOpnorm1/gamma*
_output_shapes
:*
_class
loc:@norm1/gamma*
dtype0
l

norm1/betaVarHandleOp*
shared_name
norm1/beta*
dtype0*
_output_shapes
: *
shape:
�
norm1/beta/Read/ReadVariableOpReadVariableOp
norm1/beta*
_class
loc:@norm1/beta*
dtype0*
_output_shapes
:
�
conv2_1_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:@�*!
shared_nameconv2_1_1/kernel
�
$conv2_1_1/kernel/Read/ReadVariableOpReadVariableOpconv2_1_1/kernel*#
_class
loc:@conv2_1_1/kernel*
dtype0*'
_output_shapes
:@�
u
conv2_1_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv2_1_1/bias
�
"conv2_1_1/bias/Read/ReadVariableOpReadVariableOpconv2_1_1/bias*!
_class
loc:@conv2_1_1/bias*
dtype0*
_output_shapes	
:�
�
conv2_2_1/kernelVarHandleOp*!
shared_nameconv2_2_1/kernel*
dtype0*
_output_shapes
: *
shape:��
�
$conv2_2_1/kernel/Read/ReadVariableOpReadVariableOpconv2_2_1/kernel*#
_class
loc:@conv2_2_1/kernel*
dtype0*(
_output_shapes
:��
u
conv2_2_1/biasVarHandleOp*
shape:�*
shared_nameconv2_2_1/bias*
dtype0*
_output_shapes
: 
�
"conv2_2_1/bias/Read/ReadVariableOpReadVariableOpconv2_2_1/bias*!
_class
loc:@conv2_2_1/bias*
dtype0*
_output_shapes	
:�
n
norm2/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm2/gamma
�
norm2/gamma/Read/ReadVariableOpReadVariableOpnorm2/gamma*
_class
loc:@norm2/gamma*
dtype0*
_output_shapes
:
l

norm2/betaVarHandleOp*
shape:*
shared_name
norm2/beta*
dtype0*
_output_shapes
: 
�
norm2/beta/Read/ReadVariableOpReadVariableOp
norm2/beta*
_class
loc:@norm2/beta*
dtype0*
_output_shapes
:
�
conv3_1_1/kernelVarHandleOp*
shape:��*!
shared_nameconv3_1_1/kernel*
dtype0*
_output_shapes
: 
�
$conv3_1_1/kernel/Read/ReadVariableOpReadVariableOpconv3_1_1/kernel*#
_class
loc:@conv3_1_1/kernel*
dtype0*(
_output_shapes
:��
u
conv3_1_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv3_1_1/bias
�
"conv3_1_1/bias/Read/ReadVariableOpReadVariableOpconv3_1_1/bias*
_output_shapes	
:�*!
_class
loc:@conv3_1_1/bias*
dtype0
�
conv3_2_1/kernelVarHandleOp*!
shared_nameconv3_2_1/kernel*
dtype0*
_output_shapes
: *
shape:��
�
$conv3_2_1/kernel/Read/ReadVariableOpReadVariableOpconv3_2_1/kernel*(
_output_shapes
:��*#
_class
loc:@conv3_2_1/kernel*
dtype0
u
conv3_2_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv3_2_1/bias
�
"conv3_2_1/bias/Read/ReadVariableOpReadVariableOpconv3_2_1/bias*
dtype0*
_output_shapes	
:�*!
_class
loc:@conv3_2_1/bias
n
norm3/gammaVarHandleOp*
shape:*
shared_namenorm3/gamma*
dtype0*
_output_shapes
: 
�
norm3/gamma/Read/ReadVariableOpReadVariableOpnorm3/gamma*
_class
loc:@norm3/gamma*
dtype0*
_output_shapes
:
l

norm3/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_name
norm3/beta
�
norm3/beta/Read/ReadVariableOpReadVariableOp
norm3/beta*
_class
loc:@norm3/beta*
dtype0*
_output_shapes
:
�
conv4_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��*
shared_nameconv4_1/kernel
�
"conv4_1/kernel/Read/ReadVariableOpReadVariableOpconv4_1/kernel*!
_class
loc:@conv4_1/kernel*
dtype0*(
_output_shapes
:��
q
conv4_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv4_1/bias
�
 conv4_1/bias/Read/ReadVariableOpReadVariableOpconv4_1/bias*
_output_shapes	
:�*
_class
loc:@conv4_1/bias*
dtype0
r
norm4_1/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm4_1/gamma
�
!norm4_1/gamma/Read/ReadVariableOpReadVariableOpnorm4_1/gamma* 
_class
loc:@norm4_1/gamma*
dtype0*
_output_shapes
:
p
norm4_1/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm4_1/beta
�
 norm4_1/beta/Read/ReadVariableOpReadVariableOpnorm4_1/beta*
_class
loc:@norm4_1/beta*
dtype0*
_output_shapes
:
�
conv4_2/kernelVarHandleOp*
_output_shapes
: *
shape:��*
shared_nameconv4_2/kernel*
dtype0
�
"conv4_2/kernel/Read/ReadVariableOpReadVariableOpconv4_2/kernel*!
_class
loc:@conv4_2/kernel*
dtype0*(
_output_shapes
:��
q
conv4_2/biasVarHandleOp*
shape:�*
shared_nameconv4_2/bias*
dtype0*
_output_shapes
: 
�
 conv4_2/bias/Read/ReadVariableOpReadVariableOpconv4_2/bias*
dtype0*
_output_shapes	
:�*
_class
loc:@conv4_2/bias
r
norm4_2/gammaVarHandleOp*
shared_namenorm4_2/gamma*
dtype0*
_output_shapes
: *
shape:
�
!norm4_2/gamma/Read/ReadVariableOpReadVariableOpnorm4_2/gamma* 
_class
loc:@norm4_2/gamma*
dtype0*
_output_shapes
:
p
norm4_2/betaVarHandleOp*
shared_namenorm4_2/beta*
dtype0*
_output_shapes
: *
shape:
�
 norm4_2/beta/Read/ReadVariableOpReadVariableOpnorm4_2/beta*
_class
loc:@norm4_2/beta*
dtype0*
_output_shapes
:
�
conv5_1/kernelVarHandleOp*
shared_nameconv5_1/kernel*
dtype0*
_output_shapes
: *
shape:��
�
"conv5_1/kernel/Read/ReadVariableOpReadVariableOpconv5_1/kernel*!
_class
loc:@conv5_1/kernel*
dtype0*(
_output_shapes
:��
q
conv5_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv5_1/bias
�
 conv5_1/bias/Read/ReadVariableOpReadVariableOpconv5_1/bias*
_output_shapes	
:�*
_class
loc:@conv5_1/bias*
dtype0
r
norm5_1/gammaVarHandleOp*
shape:*
shared_namenorm5_1/gamma*
dtype0*
_output_shapes
: 
�
!norm5_1/gamma/Read/ReadVariableOpReadVariableOpnorm5_1/gamma* 
_class
loc:@norm5_1/gamma*
dtype0*
_output_shapes
:
p
norm5_1/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm5_1/beta
�
 norm5_1/beta/Read/ReadVariableOpReadVariableOpnorm5_1/beta*
_class
loc:@norm5_1/beta*
dtype0*
_output_shapes
:
�
conv5_2/kernelVarHandleOp*
shared_nameconv5_2/kernel*
dtype0*
_output_shapes
: *
shape:��
�
"conv5_2/kernel/Read/ReadVariableOpReadVariableOpconv5_2/kernel*!
_class
loc:@conv5_2/kernel*
dtype0*(
_output_shapes
:��
q
conv5_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv5_2/bias
�
 conv5_2/bias/Read/ReadVariableOpReadVariableOpconv5_2/bias*
_class
loc:@conv5_2/bias*
dtype0*
_output_shapes	
:�
r
norm5_2/gammaVarHandleOp*
shape:*
shared_namenorm5_2/gamma*
dtype0*
_output_shapes
: 
�
!norm5_2/gamma/Read/ReadVariableOpReadVariableOpnorm5_2/gamma* 
_class
loc:@norm5_2/gamma*
dtype0*
_output_shapes
:
p
norm5_2/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm5_2/beta
�
 norm5_2/beta/Read/ReadVariableOpReadVariableOpnorm5_2/beta*
dtype0*
_output_shapes
:*
_class
loc:@norm5_2/beta
�
conv6_1/kernelVarHandleOp*
_output_shapes
: *
shape:��*
shared_nameconv6_1/kernel*
dtype0
�
"conv6_1/kernel/Read/ReadVariableOpReadVariableOpconv6_1/kernel*!
_class
loc:@conv6_1/kernel*
dtype0*(
_output_shapes
:��
q
conv6_1/biasVarHandleOp*
shared_nameconv6_1/bias*
dtype0*
_output_shapes
: *
shape:�
�
 conv6_1/bias/Read/ReadVariableOpReadVariableOpconv6_1/bias*
_output_shapes	
:�*
_class
loc:@conv6_1/bias*
dtype0
r
norm6_1/gammaVarHandleOp*
_output_shapes
: *
shape:*
shared_namenorm6_1/gamma*
dtype0
�
!norm6_1/gamma/Read/ReadVariableOpReadVariableOpnorm6_1/gamma* 
_class
loc:@norm6_1/gamma*
dtype0*
_output_shapes
:
p
norm6_1/betaVarHandleOp*
shape:*
shared_namenorm6_1/beta*
dtype0*
_output_shapes
: 
�
 norm6_1/beta/Read/ReadVariableOpReadVariableOpnorm6_1/beta*
_class
loc:@norm6_1/beta*
dtype0*
_output_shapes
:
�
conv6_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��*
shared_nameconv6_2/kernel
�
"conv6_2/kernel/Read/ReadVariableOpReadVariableOpconv6_2/kernel*!
_class
loc:@conv6_2/kernel*
dtype0*(
_output_shapes
:��
q
conv6_2/biasVarHandleOp*
shape:�*
shared_nameconv6_2/bias*
dtype0*
_output_shapes
: 
�
 conv6_2/bias/Read/ReadVariableOpReadVariableOpconv6_2/bias*
_class
loc:@conv6_2/bias*
dtype0*
_output_shapes	
:�
r
norm6_2/gammaVarHandleOp*
shape:*
shared_namenorm6_2/gamma*
dtype0*
_output_shapes
: 
�
!norm6_2/gamma/Read/ReadVariableOpReadVariableOpnorm6_2/gamma* 
_class
loc:@norm6_2/gamma*
dtype0*
_output_shapes
:
p
norm6_2/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm6_2/beta
�
 norm6_2/beta/Read/ReadVariableOpReadVariableOpnorm6_2/beta*
_class
loc:@norm6_2/beta*
dtype0*
_output_shapes
:
�
conv7_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��*
shared_nameconv7_1/kernel
�
"conv7_1/kernel/Read/ReadVariableOpReadVariableOpconv7_1/kernel*!
_class
loc:@conv7_1/kernel*
dtype0*(
_output_shapes
:��
q
conv7_1/biasVarHandleOp*
_output_shapes
: *
shape:�*
shared_nameconv7_1/bias*
dtype0
�
 conv7_1/bias/Read/ReadVariableOpReadVariableOpconv7_1/bias*
_output_shapes	
:�*
_class
loc:@conv7_1/bias*
dtype0
r
norm7_1/gammaVarHandleOp*
_output_shapes
: *
shape:*
shared_namenorm7_1/gamma*
dtype0
�
!norm7_1/gamma/Read/ReadVariableOpReadVariableOpnorm7_1/gamma* 
_class
loc:@norm7_1/gamma*
dtype0*
_output_shapes
:
p
norm7_1/betaVarHandleOp*
_output_shapes
: *
shape:*
shared_namenorm7_1/beta*
dtype0
�
 norm7_1/beta/Read/ReadVariableOpReadVariableOpnorm7_1/beta*
dtype0*
_output_shapes
:*
_class
loc:@norm7_1/beta
�
conv7_2/kernelVarHandleOp*
shape:��*
shared_nameconv7_2/kernel*
dtype0*
_output_shapes
: 
�
"conv7_2/kernel/Read/ReadVariableOpReadVariableOpconv7_2/kernel*!
_class
loc:@conv7_2/kernel*
dtype0*(
_output_shapes
:��
q
conv7_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv7_2/bias
�
 conv7_2/bias/Read/ReadVariableOpReadVariableOpconv7_2/bias*
dtype0*
_output_shapes	
:�*
_class
loc:@conv7_2/bias
r
norm7_2/gammaVarHandleOp*
shape:*
shared_namenorm7_2/gamma*
dtype0*
_output_shapes
: 
�
!norm7_2/gamma/Read/ReadVariableOpReadVariableOpnorm7_2/gamma* 
_class
loc:@norm7_2/gamma*
dtype0*
_output_shapes
:
p
norm7_2/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm7_2/beta
�
 norm7_2/beta/Read/ReadVariableOpReadVariableOpnorm7_2/beta*
_class
loc:@norm7_2/beta*
dtype0*
_output_shapes
:
�
conv8_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��*
shared_nameconv8_1/kernel
�
"conv8_1/kernel/Read/ReadVariableOpReadVariableOpconv8_1/kernel*(
_output_shapes
:��*!
_class
loc:@conv8_1/kernel*
dtype0
q
conv8_1/biasVarHandleOp*
shape:�*
shared_nameconv8_1/bias*
dtype0*
_output_shapes
: 
�
 conv8_1/bias/Read/ReadVariableOpReadVariableOpconv8_1/bias*
_class
loc:@conv8_1/bias*
dtype0*
_output_shapes	
:�
r
norm8_1/gammaVarHandleOp*
_output_shapes
: *
shape:*
shared_namenorm8_1/gamma*
dtype0
�
!norm8_1/gamma/Read/ReadVariableOpReadVariableOpnorm8_1/gamma* 
_class
loc:@norm8_1/gamma*
dtype0*
_output_shapes
:
p
norm8_1/betaVarHandleOp*
shared_namenorm8_1/beta*
dtype0*
_output_shapes
: *
shape:
�
 norm8_1/beta/Read/ReadVariableOpReadVariableOpnorm8_1/beta*
_output_shapes
:*
_class
loc:@norm8_1/beta*
dtype0
�
conv8_2/kernelVarHandleOp*
shared_nameconv8_2/kernel*
dtype0*
_output_shapes
: *
shape:��
�
"conv8_2/kernel/Read/ReadVariableOpReadVariableOpconv8_2/kernel*!
_class
loc:@conv8_2/kernel*
dtype0*(
_output_shapes
:��
q
conv8_2/biasVarHandleOp*
shape:�*
shared_nameconv8_2/bias*
dtype0*
_output_shapes
: 
�
 conv8_2/bias/Read/ReadVariableOpReadVariableOpconv8_2/bias*
_class
loc:@conv8_2/bias*
dtype0*
_output_shapes	
:�
r
norm8_2/gammaVarHandleOp*
_output_shapes
: *
shape:*
shared_namenorm8_2/gamma*
dtype0
�
!norm8_2/gamma/Read/ReadVariableOpReadVariableOpnorm8_2/gamma* 
_class
loc:@norm8_2/gamma*
dtype0*
_output_shapes
:
p
norm8_2/betaVarHandleOp*
shared_namenorm8_2/beta*
dtype0*
_output_shapes
: *
shape:
�
 norm8_2/beta/Read/ReadVariableOpReadVariableOpnorm8_2/beta*
_class
loc:@norm8_2/beta*
dtype0*
_output_shapes
:
�
conv9_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��*
shared_nameconv9_1/kernel
�
"conv9_1/kernel/Read/ReadVariableOpReadVariableOpconv9_1/kernel*
dtype0*(
_output_shapes
:��*!
_class
loc:@conv9_1/kernel
q
conv9_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv9_1/bias
�
 conv9_1/bias/Read/ReadVariableOpReadVariableOpconv9_1/bias*
_class
loc:@conv9_1/bias*
dtype0*
_output_shapes	
:�
r
norm9_1/gammaVarHandleOp*
shape:*
shared_namenorm9_1/gamma*
dtype0*
_output_shapes
: 
�
!norm9_1/gamma/Read/ReadVariableOpReadVariableOpnorm9_1/gamma* 
_class
loc:@norm9_1/gamma*
dtype0*
_output_shapes
:
p
norm9_1/betaVarHandleOp*
_output_shapes
: *
shape:*
shared_namenorm9_1/beta*
dtype0
�
 norm9_1/beta/Read/ReadVariableOpReadVariableOpnorm9_1/beta*
_class
loc:@norm9_1/beta*
dtype0*
_output_shapes
:
�
conv9_2/kernelVarHandleOp*
shape:��*
shared_nameconv9_2/kernel*
dtype0*
_output_shapes
: 
�
"conv9_2/kernel/Read/ReadVariableOpReadVariableOpconv9_2/kernel*!
_class
loc:@conv9_2/kernel*
dtype0*(
_output_shapes
:��
q
conv9_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv9_2/bias
�
 conv9_2/bias/Read/ReadVariableOpReadVariableOpconv9_2/bias*
_class
loc:@conv9_2/bias*
dtype0*
_output_shapes	
:�
r
norm9_2/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm9_2/gamma
�
!norm9_2/gamma/Read/ReadVariableOpReadVariableOpnorm9_2/gamma* 
_class
loc:@norm9_2/gamma*
dtype0*
_output_shapes
:
p
norm9_2/betaVarHandleOp*
shared_namenorm9_2/beta*
dtype0*
_output_shapes
: *
shape:
�
 norm9_2/beta/Read/ReadVariableOpReadVariableOpnorm9_2/beta*
_class
loc:@norm9_2/beta*
dtype0*
_output_shapes
:
�
conv10_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��* 
shared_nameconv10_1/kernel
�
#conv10_1/kernel/Read/ReadVariableOpReadVariableOpconv10_1/kernel*
dtype0*(
_output_shapes
:��*"
_class
loc:@conv10_1/kernel
s
conv10_1/biasVarHandleOp*
shape:�*
shared_nameconv10_1/bias*
dtype0*
_output_shapes
: 
�
!conv10_1/bias/Read/ReadVariableOpReadVariableOpconv10_1/bias* 
_class
loc:@conv10_1/bias*
dtype0*
_output_shapes	
:�
t
norm10_1/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm10_1/gamma
�
"norm10_1/gamma/Read/ReadVariableOpReadVariableOpnorm10_1/gamma*!
_class
loc:@norm10_1/gamma*
dtype0*
_output_shapes
:
r
norm10_1/betaVarHandleOp*
shared_namenorm10_1/beta*
dtype0*
_output_shapes
: *
shape:
�
!norm10_1/beta/Read/ReadVariableOpReadVariableOpnorm10_1/beta* 
_class
loc:@norm10_1/beta*
dtype0*
_output_shapes
:
�
conv10_2/kernelVarHandleOp* 
shared_nameconv10_2/kernel*
dtype0*
_output_shapes
: *
shape:��
�
#conv10_2/kernel/Read/ReadVariableOpReadVariableOpconv10_2/kernel*"
_class
loc:@conv10_2/kernel*
dtype0*(
_output_shapes
:��
s
conv10_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv10_2/bias
�
!conv10_2/bias/Read/ReadVariableOpReadVariableOpconv10_2/bias* 
_class
loc:@conv10_2/bias*
dtype0*
_output_shapes	
:�
t
norm10_2/gammaVarHandleOp*
_output_shapes
: *
shape:*
shared_namenorm10_2/gamma*
dtype0
�
"norm10_2/gamma/Read/ReadVariableOpReadVariableOpnorm10_2/gamma*!
_class
loc:@norm10_2/gamma*
dtype0*
_output_shapes
:
r
norm10_2/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm10_2/beta
�
!norm10_2/beta/Read/ReadVariableOpReadVariableOpnorm10_2/beta*
dtype0*
_output_shapes
:* 
_class
loc:@norm10_2/beta
�
conv11_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��* 
shared_nameconv11_1/kernel
�
#conv11_1/kernel/Read/ReadVariableOpReadVariableOpconv11_1/kernel*"
_class
loc:@conv11_1/kernel*
dtype0*(
_output_shapes
:��
s
conv11_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv11_1/bias
�
!conv11_1/bias/Read/ReadVariableOpReadVariableOpconv11_1/bias* 
_class
loc:@conv11_1/bias*
dtype0*
_output_shapes	
:�
t
norm11_1/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm11_1/gamma
�
"norm11_1/gamma/Read/ReadVariableOpReadVariableOpnorm11_1/gamma*!
_class
loc:@norm11_1/gamma*
dtype0*
_output_shapes
:
r
norm11_1/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm11_1/beta
�
!norm11_1/beta/Read/ReadVariableOpReadVariableOpnorm11_1/beta* 
_class
loc:@norm11_1/beta*
dtype0*
_output_shapes
:
�
conv11_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��* 
shared_nameconv11_2/kernel
�
#conv11_2/kernel/Read/ReadVariableOpReadVariableOpconv11_2/kernel*"
_class
loc:@conv11_2/kernel*
dtype0*(
_output_shapes
:��
s
conv11_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv11_2/bias
�
!conv11_2/bias/Read/ReadVariableOpReadVariableOpconv11_2/bias* 
_class
loc:@conv11_2/bias*
dtype0*
_output_shapes	
:�
t
norm11_2/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namenorm11_2/gamma
�
"norm11_2/gamma/Read/ReadVariableOpReadVariableOpnorm11_2/gamma*!
_class
loc:@norm11_2/gamma*
dtype0*
_output_shapes
:
r
norm11_2/betaVarHandleOp*
shared_namenorm11_2/beta*
dtype0*
_output_shapes
: *
shape:
�
!norm11_2/beta/Read/ReadVariableOpReadVariableOpnorm11_2/beta* 
_class
loc:@norm11_2/beta*
dtype0*
_output_shapes
:
�
deconv1_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��*!
shared_namedeconv1_1/kernel
�
$deconv1_1/kernel/Read/ReadVariableOpReadVariableOpdeconv1_1/kernel*
dtype0*(
_output_shapes
:��*#
_class
loc:@deconv1_1/kernel
u
deconv1_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_namedeconv1_1/bias
�
"deconv1_1/bias/Read/ReadVariableOpReadVariableOpdeconv1_1/bias*!
_class
loc:@deconv1_1/bias*
dtype0*
_output_shapes	
:�
�
deconv1_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��*!
shared_namedeconv1_2/kernel
�
$deconv1_2/kernel/Read/ReadVariableOpReadVariableOpdeconv1_2/kernel*#
_class
loc:@deconv1_2/kernel*
dtype0*(
_output_shapes
:��
u
deconv1_2/biasVarHandleOp*
shared_namedeconv1_2/bias*
dtype0*
_output_shapes
: *
shape:�
�
"deconv1_2/bias/Read/ReadVariableOpReadVariableOpdeconv1_2/bias*!
_class
loc:@deconv1_2/bias*
dtype0*
_output_shapes	
:�
|
norm_deconv1/gammaVarHandleOp*#
shared_namenorm_deconv1/gamma*
dtype0*
_output_shapes
: *
shape:
�
&norm_deconv1/gamma/Read/ReadVariableOpReadVariableOpnorm_deconv1/gamma*
_output_shapes
:*%
_class
loc:@norm_deconv1/gamma*
dtype0
z
norm_deconv1/betaVarHandleOp*"
shared_namenorm_deconv1/beta*
dtype0*
_output_shapes
: *
shape:
�
%norm_deconv1/beta/Read/ReadVariableOpReadVariableOpnorm_deconv1/beta*$
_class
loc:@norm_deconv1/beta*
dtype0*
_output_shapes
:
�
deconv2_1/kernelVarHandleOp*!
shared_namedeconv2_1/kernel*
dtype0*
_output_shapes
: *
shape:��
�
$deconv2_1/kernel/Read/ReadVariableOpReadVariableOpdeconv2_1/kernel*#
_class
loc:@deconv2_1/kernel*
dtype0*(
_output_shapes
:��
u
deconv2_1/biasVarHandleOp*
_output_shapes
: *
shape:�*
shared_namedeconv2_1/bias*
dtype0
�
"deconv2_1/bias/Read/ReadVariableOpReadVariableOpdeconv2_1/bias*!
_class
loc:@deconv2_1/bias*
dtype0*
_output_shapes	
:�
�
deconv2_2/kernelVarHandleOp*
shape:��*!
shared_namedeconv2_2/kernel*
dtype0*
_output_shapes
: 
�
$deconv2_2/kernel/Read/ReadVariableOpReadVariableOpdeconv2_2/kernel*#
_class
loc:@deconv2_2/kernel*
dtype0*(
_output_shapes
:��
u
deconv2_2/biasVarHandleOp*
shape:�*
shared_namedeconv2_2/bias*
dtype0*
_output_shapes
: 
�
"deconv2_2/bias/Read/ReadVariableOpReadVariableOpdeconv2_2/bias*!
_class
loc:@deconv2_2/bias*
dtype0*
_output_shapes	
:�
|
norm_deconv2/gammaVarHandleOp*#
shared_namenorm_deconv2/gamma*
dtype0*
_output_shapes
: *
shape:
�
&norm_deconv2/gamma/Read/ReadVariableOpReadVariableOpnorm_deconv2/gamma*%
_class
loc:@norm_deconv2/gamma*
dtype0*
_output_shapes
:
z
norm_deconv2/betaVarHandleOp*"
shared_namenorm_deconv2/beta*
dtype0*
_output_shapes
: *
shape:
�
%norm_deconv2/beta/Read/ReadVariableOpReadVariableOpnorm_deconv2/beta*
_output_shapes
:*$
_class
loc:@norm_deconv2/beta*
dtype0
�
deconv3/kernelVarHandleOp*
shape:�*
shared_namedeconv3/kernel*
dtype0*
_output_shapes
: 
�
"deconv3/kernel/Read/ReadVariableOpReadVariableOpdeconv3/kernel*!
_class
loc:@deconv3/kernel*
dtype0*'
_output_shapes
:�
p
deconv3/biasVarHandleOp*
shape:*
shared_namedeconv3/bias*
dtype0*
_output_shapes
: 
�
 deconv3/bias/Read/ReadVariableOpReadVariableOpdeconv3/bias*
_class
loc:@deconv3/bias*
dtype0*
_output_shapes
:

NoOpNoOp
�`
ConstConst"/device:CPU:0*
_output_shapes
: *�`
value�`B�` B�`
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer_with_weights-14
layer-26
layer_with_weights-15
layer-27
layer-28
layer-29
layer_with_weights-16
layer-30
 layer_with_weights-17
 layer-31
!layer-32
"layer-33
#layer_with_weights-18
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer_with_weights-21
(layer-39
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,layer_with_weights-23
,layer-43
-layer-44
.layer-45
/layer_with_weights-24
/layer-46
0layer_with_weights-25
0layer-47
1layer-48
2layer-49
3layer_with_weights-26
3layer-50
4layer_with_weights-27
4layer-51
5layer-52
6layer-53
7layer_with_weights-28
7layer-54
8layer_with_weights-29
8layer-55
9layer-56
:layer-57
;layer_with_weights-30
;layer-58
<layer_with_weights-31
<layer-59
=layer-60
>layer-61
?layer_with_weights-32
?layer-62
@layer_with_weights-33
@layer-63
Alayer-64
Blayer-65
Clayer_with_weights-34
Clayer-66
Dlayer_with_weights-35
Dlayer-67
Elayer-68
Flayer-69
Glayer_with_weights-36
Glayer-70
Hlayer_with_weights-37
Hlayer-71
Ilayer-72
Jlayer-73
Klayer_with_weights-38
Klayer-74
Llayer_with_weights-39
Llayer-75
Mlayer-76
Nlayer_with_weights-40
Nlayer-77
Olayer_with_weights-41
Olayer-78
Player_with_weights-42
Player-79
Qlayer-80
Rlayer_with_weights-43
Rlayer-81
Slayer_with_weights-44
Slayer-82
Tlayer_with_weights-45
Tlayer-83
Ulayer-84
Vlayer-85
Wlayer_with_weights-46
Wlayer-86
Xlayer-87
Y
signatures
 
 


Zkernel
[bias

	\gamma
]beta
 


^kernel
_bias


`kernel
abias

	bgamma
cbeta
 


dkernel
ebias


fkernel
gbias

	hgamma
ibeta
 
 


jkernel
kbias

	lgamma
mbeta
 
 


nkernel
obias

	pgamma
qbeta
 
 


rkernel
sbias

	tgamma
ubeta
 
 


vkernel
wbias

	xgamma
ybeta
 
 


zkernel
{bias

	|gamma
}beta
 
 


~kernel
bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias


�gamma
	�beta
 

�kernel
	�bias

�kernel
	�bias


�gamma
	�beta
 

�kernel
	�bias

�kernel
	�bias


�gamma
	�beta
 
 

�kernel
	�bias
 
 
ZX
VARIABLE_VALUEconv1_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEnorm1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
norm1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv2_1_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2_1_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv2_2_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2_2_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEnorm2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
norm2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv3_1_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3_1_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv3_2_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3_2_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEnorm3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
norm3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv4_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv4_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnorm4_1/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEnorm4_1/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv4_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv4_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm4_2/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm4_2/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv5_1/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv5_1/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm5_1/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm5_1/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv5_2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv5_2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm5_2/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm5_2/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv6_1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv6_1/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm6_1/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm6_1/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv6_2/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv6_2/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm6_2/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm6_2/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv7_1/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv7_1/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm7_1/gamma6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm7_1/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv7_2/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv7_2/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm7_2/gamma6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm7_2/beta5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv8_1/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv8_1/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm8_1/gamma6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm8_1/beta5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv8_2/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv8_2/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm8_2/gamma6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm8_2/beta5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv9_1/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv9_1/bias5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm9_1/gamma6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm9_1/beta5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv9_2/kernel7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv9_2/bias5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnorm9_2/gamma6layer_with_weights-31/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnorm9_2/beta5layer_with_weights-31/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv10_1/kernel7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv10_1/bias5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEnorm10_1/gamma6layer_with_weights-33/gamma/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnorm10_1/beta5layer_with_weights-33/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv10_2/kernel7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv10_2/bias5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEnorm10_2/gamma6layer_with_weights-35/gamma/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnorm10_2/beta5layer_with_weights-35/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv11_1/kernel7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv11_1/bias5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEnorm11_1/gamma6layer_with_weights-37/gamma/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnorm11_1/beta5layer_with_weights-37/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv11_2/kernel7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv11_2/bias5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEnorm11_2/gamma6layer_with_weights-39/gamma/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnorm11_2/beta5layer_with_weights-39/beta/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdeconv1_1/kernel7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdeconv1_1/bias5layer_with_weights-40/bias/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdeconv1_2/kernel7layer_with_weights-41/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdeconv1_2/bias5layer_with_weights-41/bias/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEnorm_deconv1/gamma6layer_with_weights-42/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEnorm_deconv1/beta5layer_with_weights-42/beta/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdeconv2_1/kernel7layer_with_weights-43/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdeconv2_1/bias5layer_with_weights-43/bias/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdeconv2_2/kernel7layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdeconv2_2/bias5layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEnorm_deconv2/gamma6layer_with_weights-45/gamma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEnorm_deconv2/beta5layer_with_weights-45/beta/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdeconv3/kernel7layer_with_weights-46/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdeconv3/bias5layer_with_weights-46/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
�
serving_default_inputPlaceholder*
dtype0*1
_output_shapes
:�����������*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv1_1/kernelconv1_1/biasnorm1/gamma
norm1/betaconv2_1_1/kernelconv2_1_1/biasconv2_2_1/kernelconv2_2_1/biasnorm2/gamma
norm2/betaconv3_1_1/kernelconv3_1_1/biasconv3_2_1/kernelconv3_2_1/biasnorm3/gamma
norm3/betaconv4_1/kernelconv4_1/biasnorm4_1/gammanorm4_1/betaconv4_2/kernelconv4_2/biasnorm4_2/gammanorm4_2/betaconv5_1/kernelconv5_1/biasnorm5_1/gammanorm5_1/betaconv5_2/kernelconv5_2/biasnorm5_2/gammanorm5_2/betaconv6_1/kernelconv6_1/biasnorm6_1/gammanorm6_1/betaconv6_2/kernelconv6_2/biasnorm6_2/gammanorm6_2/betaconv7_1/kernelconv7_1/biasnorm7_1/gammanorm7_1/betaconv7_2/kernelconv7_2/biasnorm7_2/gammanorm7_2/betaconv8_1/kernelconv8_1/biasnorm8_1/gammanorm8_1/betaconv8_2/kernelconv8_2/biasnorm8_2/gammanorm8_2/betaconv9_1/kernelconv9_1/biasnorm9_1/gammanorm9_1/betaconv9_2/kernelconv9_2/biasnorm9_2/gammanorm9_2/betaconv10_1/kernelconv10_1/biasnorm10_1/gammanorm10_1/betaconv10_2/kernelconv10_2/biasnorm10_2/gammanorm10_2/betaconv11_1/kernelconv11_1/biasnorm11_1/gammanorm11_1/betaconv11_2/kernelconv11_2/biasnorm11_2/gammanorm11_2/betadeconv1_1/kerneldeconv1_1/biasdeconv1_2/kerneldeconv1_2/biasnorm_deconv1/gammanorm_deconv1/betadeconv2_1/kerneldeconv2_1/biasdeconv2_2/kerneldeconv2_2/biasnorm_deconv2/gammanorm_deconv2/betadeconv3/kerneldeconv3/bias**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*j
Tinc
a2_*+
f&R$
"__inference_signature_wrapper_7276*
Tout
2
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"conv1_1/kernel/Read/ReadVariableOp conv1_1/bias/Read/ReadVariableOpnorm1/gamma/Read/ReadVariableOpnorm1/beta/Read/ReadVariableOp$conv2_1_1/kernel/Read/ReadVariableOp"conv2_1_1/bias/Read/ReadVariableOp$conv2_2_1/kernel/Read/ReadVariableOp"conv2_2_1/bias/Read/ReadVariableOpnorm2/gamma/Read/ReadVariableOpnorm2/beta/Read/ReadVariableOp$conv3_1_1/kernel/Read/ReadVariableOp"conv3_1_1/bias/Read/ReadVariableOp$conv3_2_1/kernel/Read/ReadVariableOp"conv3_2_1/bias/Read/ReadVariableOpnorm3/gamma/Read/ReadVariableOpnorm3/beta/Read/ReadVariableOp"conv4_1/kernel/Read/ReadVariableOp conv4_1/bias/Read/ReadVariableOp!norm4_1/gamma/Read/ReadVariableOp norm4_1/beta/Read/ReadVariableOp"conv4_2/kernel/Read/ReadVariableOp conv4_2/bias/Read/ReadVariableOp!norm4_2/gamma/Read/ReadVariableOp norm4_2/beta/Read/ReadVariableOp"conv5_1/kernel/Read/ReadVariableOp conv5_1/bias/Read/ReadVariableOp!norm5_1/gamma/Read/ReadVariableOp norm5_1/beta/Read/ReadVariableOp"conv5_2/kernel/Read/ReadVariableOp conv5_2/bias/Read/ReadVariableOp!norm5_2/gamma/Read/ReadVariableOp norm5_2/beta/Read/ReadVariableOp"conv6_1/kernel/Read/ReadVariableOp conv6_1/bias/Read/ReadVariableOp!norm6_1/gamma/Read/ReadVariableOp norm6_1/beta/Read/ReadVariableOp"conv6_2/kernel/Read/ReadVariableOp conv6_2/bias/Read/ReadVariableOp!norm6_2/gamma/Read/ReadVariableOp norm6_2/beta/Read/ReadVariableOp"conv7_1/kernel/Read/ReadVariableOp conv7_1/bias/Read/ReadVariableOp!norm7_1/gamma/Read/ReadVariableOp norm7_1/beta/Read/ReadVariableOp"conv7_2/kernel/Read/ReadVariableOp conv7_2/bias/Read/ReadVariableOp!norm7_2/gamma/Read/ReadVariableOp norm7_2/beta/Read/ReadVariableOp"conv8_1/kernel/Read/ReadVariableOp conv8_1/bias/Read/ReadVariableOp!norm8_1/gamma/Read/ReadVariableOp norm8_1/beta/Read/ReadVariableOp"conv8_2/kernel/Read/ReadVariableOp conv8_2/bias/Read/ReadVariableOp!norm8_2/gamma/Read/ReadVariableOp norm8_2/beta/Read/ReadVariableOp"conv9_1/kernel/Read/ReadVariableOp conv9_1/bias/Read/ReadVariableOp!norm9_1/gamma/Read/ReadVariableOp norm9_1/beta/Read/ReadVariableOp"conv9_2/kernel/Read/ReadVariableOp conv9_2/bias/Read/ReadVariableOp!norm9_2/gamma/Read/ReadVariableOp norm9_2/beta/Read/ReadVariableOp#conv10_1/kernel/Read/ReadVariableOp!conv10_1/bias/Read/ReadVariableOp"norm10_1/gamma/Read/ReadVariableOp!norm10_1/beta/Read/ReadVariableOp#conv10_2/kernel/Read/ReadVariableOp!conv10_2/bias/Read/ReadVariableOp"norm10_2/gamma/Read/ReadVariableOp!norm10_2/beta/Read/ReadVariableOp#conv11_1/kernel/Read/ReadVariableOp!conv11_1/bias/Read/ReadVariableOp"norm11_1/gamma/Read/ReadVariableOp!norm11_1/beta/Read/ReadVariableOp#conv11_2/kernel/Read/ReadVariableOp!conv11_2/bias/Read/ReadVariableOp"norm11_2/gamma/Read/ReadVariableOp!norm11_2/beta/Read/ReadVariableOp$deconv1_1/kernel/Read/ReadVariableOp"deconv1_1/bias/Read/ReadVariableOp$deconv1_2/kernel/Read/ReadVariableOp"deconv1_2/bias/Read/ReadVariableOp&norm_deconv1/gamma/Read/ReadVariableOp%norm_deconv1/beta/Read/ReadVariableOp$deconv2_1/kernel/Read/ReadVariableOp"deconv2_1/bias/Read/ReadVariableOp$deconv2_2/kernel/Read/ReadVariableOp"deconv2_2/bias/Read/ReadVariableOp&norm_deconv2/gamma/Read/ReadVariableOp%norm_deconv2/beta/Read/ReadVariableOp"deconv3/kernel/Read/ReadVariableOp deconv3/bias/Read/ReadVariableOpConst*+
_gradient_op_typePartitionedCall-7585*&
f!R
__inference__traced_save_7584*
Tout
2**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *k
Tind
b2`
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1_1/kernelconv1_1/biasnorm1/gamma
norm1/betaconv2_1_1/kernelconv2_1_1/biasconv2_2_1/kernelconv2_2_1/biasnorm2/gamma
norm2/betaconv3_1_1/kernelconv3_1_1/biasconv3_2_1/kernelconv3_2_1/biasnorm3/gamma
norm3/betaconv4_1/kernelconv4_1/biasnorm4_1/gammanorm4_1/betaconv4_2/kernelconv4_2/biasnorm4_2/gammanorm4_2/betaconv5_1/kernelconv5_1/biasnorm5_1/gammanorm5_1/betaconv5_2/kernelconv5_2/biasnorm5_2/gammanorm5_2/betaconv6_1/kernelconv6_1/biasnorm6_1/gammanorm6_1/betaconv6_2/kernelconv6_2/biasnorm6_2/gammanorm6_2/betaconv7_1/kernelconv7_1/biasnorm7_1/gammanorm7_1/betaconv7_2/kernelconv7_2/biasnorm7_2/gammanorm7_2/betaconv8_1/kernelconv8_1/biasnorm8_1/gammanorm8_1/betaconv8_2/kernelconv8_2/biasnorm8_2/gammanorm8_2/betaconv9_1/kernelconv9_1/biasnorm9_1/gammanorm9_1/betaconv9_2/kernelconv9_2/biasnorm9_2/gammanorm9_2/betaconv10_1/kernelconv10_1/biasnorm10_1/gammanorm10_1/betaconv10_2/kernelconv10_2/biasnorm10_2/gammanorm10_2/betaconv11_1/kernelconv11_1/biasnorm11_1/gammanorm11_1/betaconv11_2/kernelconv11_2/biasnorm11_2/gammanorm11_2/betadeconv1_1/kerneldeconv1_1/biasdeconv1_2/kerneldeconv1_2/biasnorm_deconv1/gammanorm_deconv1/betadeconv2_1/kerneldeconv2_1/biasdeconv2_2/kerneldeconv2_2/biasnorm_deconv2/gammanorm_deconv2/betadeconv3/kerneldeconv3/bias*+
_gradient_op_typePartitionedCall-7880*)
f$R"
 __inference__traced_restore_7879*
Tout
2**
config_proto

CPU

GPU 2J 8*j
Tinc
a2_*
_output_shapes
: ˯
��
�-
 __inference__traced_restore_7879
file_prefix#
assignvariableop_conv1_1_kernel#
assignvariableop_1_conv1_1_bias"
assignvariableop_2_norm1_gamma!
assignvariableop_3_norm1_beta'
#assignvariableop_4_conv2_1_1_kernel%
!assignvariableop_5_conv2_1_1_bias'
#assignvariableop_6_conv2_2_1_kernel%
!assignvariableop_7_conv2_2_1_bias"
assignvariableop_8_norm2_gamma!
assignvariableop_9_norm2_beta(
$assignvariableop_10_conv3_1_1_kernel&
"assignvariableop_11_conv3_1_1_bias(
$assignvariableop_12_conv3_2_1_kernel&
"assignvariableop_13_conv3_2_1_bias#
assignvariableop_14_norm3_gamma"
assignvariableop_15_norm3_beta&
"assignvariableop_16_conv4_1_kernel$
 assignvariableop_17_conv4_1_bias%
!assignvariableop_18_norm4_1_gamma$
 assignvariableop_19_norm4_1_beta&
"assignvariableop_20_conv4_2_kernel$
 assignvariableop_21_conv4_2_bias%
!assignvariableop_22_norm4_2_gamma$
 assignvariableop_23_norm4_2_beta&
"assignvariableop_24_conv5_1_kernel$
 assignvariableop_25_conv5_1_bias%
!assignvariableop_26_norm5_1_gamma$
 assignvariableop_27_norm5_1_beta&
"assignvariableop_28_conv5_2_kernel$
 assignvariableop_29_conv5_2_bias%
!assignvariableop_30_norm5_2_gamma$
 assignvariableop_31_norm5_2_beta&
"assignvariableop_32_conv6_1_kernel$
 assignvariableop_33_conv6_1_bias%
!assignvariableop_34_norm6_1_gamma$
 assignvariableop_35_norm6_1_beta&
"assignvariableop_36_conv6_2_kernel$
 assignvariableop_37_conv6_2_bias%
!assignvariableop_38_norm6_2_gamma$
 assignvariableop_39_norm6_2_beta&
"assignvariableop_40_conv7_1_kernel$
 assignvariableop_41_conv7_1_bias%
!assignvariableop_42_norm7_1_gamma$
 assignvariableop_43_norm7_1_beta&
"assignvariableop_44_conv7_2_kernel$
 assignvariableop_45_conv7_2_bias%
!assignvariableop_46_norm7_2_gamma$
 assignvariableop_47_norm7_2_beta&
"assignvariableop_48_conv8_1_kernel$
 assignvariableop_49_conv8_1_bias%
!assignvariableop_50_norm8_1_gamma$
 assignvariableop_51_norm8_1_beta&
"assignvariableop_52_conv8_2_kernel$
 assignvariableop_53_conv8_2_bias%
!assignvariableop_54_norm8_2_gamma$
 assignvariableop_55_norm8_2_beta&
"assignvariableop_56_conv9_1_kernel$
 assignvariableop_57_conv9_1_bias%
!assignvariableop_58_norm9_1_gamma$
 assignvariableop_59_norm9_1_beta&
"assignvariableop_60_conv9_2_kernel$
 assignvariableop_61_conv9_2_bias%
!assignvariableop_62_norm9_2_gamma$
 assignvariableop_63_norm9_2_beta'
#assignvariableop_64_conv10_1_kernel%
!assignvariableop_65_conv10_1_bias&
"assignvariableop_66_norm10_1_gamma%
!assignvariableop_67_norm10_1_beta'
#assignvariableop_68_conv10_2_kernel%
!assignvariableop_69_conv10_2_bias&
"assignvariableop_70_norm10_2_gamma%
!assignvariableop_71_norm10_2_beta'
#assignvariableop_72_conv11_1_kernel%
!assignvariableop_73_conv11_1_bias&
"assignvariableop_74_norm11_1_gamma%
!assignvariableop_75_norm11_1_beta'
#assignvariableop_76_conv11_2_kernel%
!assignvariableop_77_conv11_2_bias&
"assignvariableop_78_norm11_2_gamma%
!assignvariableop_79_norm11_2_beta(
$assignvariableop_80_deconv1_1_kernel&
"assignvariableop_81_deconv1_1_bias(
$assignvariableop_82_deconv1_2_kernel&
"assignvariableop_83_deconv1_2_bias*
&assignvariableop_84_norm_deconv1_gamma)
%assignvariableop_85_norm_deconv1_beta(
$assignvariableop_86_deconv2_1_kernel&
"assignvariableop_87_deconv2_1_bias(
$assignvariableop_88_deconv2_2_kernel&
"assignvariableop_89_deconv2_2_bias*
&assignvariableop_90_norm_deconv2_gamma)
%assignvariableop_91_norm_deconv2_beta&
"assignvariableop_92_deconv3_kernel$
 assignvariableop_93_deconv3_bias
identity_95��AssignVariableOp_3�AssignVariableOp_22�AssignVariableOp_41�AssignVariableOp_60�AssignVariableOp_79�AssignVariableOp_7�AssignVariableOp_26�AssignVariableOp_45�AssignVariableOp_64�AssignVariableOp_83�AssignVariableOp_11�AssignVariableOp_30�AssignVariableOp_49�AssignVariableOp_68�AssignVariableOp_87�AssignVariableOp_15�AssignVariableOp_34�AssignVariableOp_53�AssignVariableOp_72�AssignVariableOp�AssignVariableOp_91�AssignVariableOp_19�AssignVariableOp_38�AssignVariableOp_57�AssignVariableOp_76�AssignVariableOp_4�AssignVariableOp_23�AssignVariableOp_42�AssignVariableOp_61�AssignVariableOp_80�AssignVariableOp_8�AssignVariableOp_27�AssignVariableOp_46�AssignVariableOp_65�AssignVariableOp_84�AssignVariableOp_12�AssignVariableOp_31�AssignVariableOp_50�AssignVariableOp_69�AssignVariableOp_88�AssignVariableOp_16�AssignVariableOp_35�AssignVariableOp_54�AssignVariableOp_73�AssignVariableOp_1�AssignVariableOp_92�AssignVariableOp_39�AssignVariableOp_20�AssignVariableOp_58�AssignVariableOp_77�AssignVariableOp_5�AssignVariableOp_24�AssignVariableOp_43�AssignVariableOp_62�AssignVariableOp_81�AssignVariableOp_9�AssignVariableOp_28�AssignVariableOp_47�AssignVariableOp_66�	RestoreV2�AssignVariableOp_85�AssignVariableOp_13�AssignVariableOp_32�AssignVariableOp_51�AssignVariableOp_70�AssignVariableOp_89�AssignVariableOp_17�AssignVariableOp_36�AssignVariableOp_55�AssignVariableOp_74�AssignVariableOp_2�AssignVariableOp_21�AssignVariableOp_93�AssignVariableOp_40�AssignVariableOp_59�AssignVariableOp_78�AssignVariableOp_6�AssignVariableOp_25�AssignVariableOp_44�AssignVariableOp_63�AssignVariableOp_82�AssignVariableOp_10�AssignVariableOp_29�AssignVariableOp_48�AssignVariableOp_67�AssignVariableOp_86�AssignVariableOp_14�AssignVariableOp_33�AssignVariableOp_52�AssignVariableOp_71�AssignVariableOp_90�AssignVariableOp_18�AssignVariableOp_37�RestoreV2_1�AssignVariableOp_56�AssignVariableOp_75�)
RestoreV2/tensor_namesConst"/device:CPU:0*�(
value�(B�(^B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-31/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-33/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-35/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-37/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-39/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-39/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-40/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-41/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-41/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-42/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-42/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-43/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-43/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-45/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-45/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-46/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-46/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:^�
RestoreV2/shape_and_slicesConst"/device:CPU:0*�
value�B�^B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:^�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*l
dtypesb
`2^L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:{
AssignVariableOpAssignVariableOpassignvariableop_conv1_1_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:~
AssignVariableOp_2AssignVariableOpassignvariableop_2_norm1_gammaIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:}
AssignVariableOp_3AssignVariableOpassignvariableop_3_norm1_betaIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2_1_1_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2_1_1_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2_2_1_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2_2_1_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:~
AssignVariableOp_8AssignVariableOpassignvariableop_8_norm2_gammaIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:}
AssignVariableOp_9AssignVariableOpassignvariableop_9_norm2_betaIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv3_1_1_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv3_1_1_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv3_2_1_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv3_2_1_biasIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_norm3_gammaIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_norm3_betaIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv4_1_kernelIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0�
AssignVariableOp_17AssignVariableOp assignvariableop_17_conv4_1_biasIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_norm4_1_gammaIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0�
AssignVariableOp_19AssignVariableOp assignvariableop_19_norm4_1_betaIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_conv4_2_kernelIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp assignvariableop_21_conv4_2_biasIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp!assignvariableop_22_norm4_2_gammaIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0�
AssignVariableOp_23AssignVariableOp assignvariableop_23_norm4_2_betaIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_conv5_1_kernelIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp assignvariableop_25_conv5_1_biasIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp!assignvariableop_26_norm5_1_gammaIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp assignvariableop_27_norm5_1_betaIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp"assignvariableop_28_conv5_2_kernelIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp assignvariableop_29_conv5_2_biasIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp!assignvariableop_30_norm5_2_gammaIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp assignvariableop_31_norm5_2_betaIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp"assignvariableop_32_conv6_1_kernelIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp assignvariableop_33_conv6_1_biasIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp!assignvariableop_34_norm6_1_gammaIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp assignvariableop_35_norm6_1_betaIdentity_35:output:0*
_output_shapes
 *
dtype0P
Identity_36IdentityRestoreV2:tensors:36*
_output_shapes
:*
T0�
AssignVariableOp_36AssignVariableOp"assignvariableop_36_conv6_2_kernelIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp assignvariableop_37_conv6_2_biasIdentity_37:output:0*
_output_shapes
 *
dtype0P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp!assignvariableop_38_norm6_2_gammaIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
_output_shapes
:*
T0�
AssignVariableOp_39AssignVariableOp assignvariableop_39_norm6_2_betaIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
_output_shapes
:*
T0�
AssignVariableOp_40AssignVariableOp"assignvariableop_40_conv7_1_kernelIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp assignvariableop_41_conv7_1_biasIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
_output_shapes
:*
T0�
AssignVariableOp_42AssignVariableOp!assignvariableop_42_norm7_1_gammaIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp assignvariableop_43_norm7_1_betaIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp"assignvariableop_44_conv7_2_kernelIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp assignvariableop_45_conv7_2_biasIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp!assignvariableop_46_norm7_2_gammaIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp assignvariableop_47_norm7_2_betaIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
_output_shapes
:*
T0�
AssignVariableOp_48AssignVariableOp"assignvariableop_48_conv8_1_kernelIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp assignvariableop_49_conv8_1_biasIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp!assignvariableop_50_norm8_1_gammaIdentity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp assignvariableop_51_norm8_1_betaIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
_output_shapes
:*
T0�
AssignVariableOp_52AssignVariableOp"assignvariableop_52_conv8_2_kernelIdentity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
_output_shapes
:*
T0�
AssignVariableOp_53AssignVariableOp assignvariableop_53_conv8_2_biasIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp!assignvariableop_54_norm8_2_gammaIdentity_54:output:0*
dtype0*
_output_shapes
 P
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp assignvariableop_55_norm8_2_betaIdentity_55:output:0*
dtype0*
_output_shapes
 P
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp"assignvariableop_56_conv9_1_kernelIdentity_56:output:0*
dtype0*
_output_shapes
 P
Identity_57IdentityRestoreV2:tensors:57*
_output_shapes
:*
T0�
AssignVariableOp_57AssignVariableOp assignvariableop_57_conv9_1_biasIdentity_57:output:0*
dtype0*
_output_shapes
 P
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp!assignvariableop_58_norm9_1_gammaIdentity_58:output:0*
dtype0*
_output_shapes
 P
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp assignvariableop_59_norm9_1_betaIdentity_59:output:0*
dtype0*
_output_shapes
 P
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp"assignvariableop_60_conv9_2_kernelIdentity_60:output:0*
dtype0*
_output_shapes
 P
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp assignvariableop_61_conv9_2_biasIdentity_61:output:0*
dtype0*
_output_shapes
 P
Identity_62IdentityRestoreV2:tensors:62*
_output_shapes
:*
T0�
AssignVariableOp_62AssignVariableOp!assignvariableop_62_norm9_2_gammaIdentity_62:output:0*
dtype0*
_output_shapes
 P
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp assignvariableop_63_norm9_2_betaIdentity_63:output:0*
dtype0*
_output_shapes
 P
Identity_64IdentityRestoreV2:tensors:64*
_output_shapes
:*
T0�
AssignVariableOp_64AssignVariableOp#assignvariableop_64_conv10_1_kernelIdentity_64:output:0*
dtype0*
_output_shapes
 P
Identity_65IdentityRestoreV2:tensors:65*
_output_shapes
:*
T0�
AssignVariableOp_65AssignVariableOp!assignvariableop_65_conv10_1_biasIdentity_65:output:0*
_output_shapes
 *
dtype0P
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp"assignvariableop_66_norm10_1_gammaIdentity_66:output:0*
dtype0*
_output_shapes
 P
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp!assignvariableop_67_norm10_1_betaIdentity_67:output:0*
dtype0*
_output_shapes
 P
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp#assignvariableop_68_conv10_2_kernelIdentity_68:output:0*
dtype0*
_output_shapes
 P
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp!assignvariableop_69_conv10_2_biasIdentity_69:output:0*
dtype0*
_output_shapes
 P
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp"assignvariableop_70_norm10_2_gammaIdentity_70:output:0*
dtype0*
_output_shapes
 P
Identity_71IdentityRestoreV2:tensors:71*
_output_shapes
:*
T0�
AssignVariableOp_71AssignVariableOp!assignvariableop_71_norm10_2_betaIdentity_71:output:0*
dtype0*
_output_shapes
 P
Identity_72IdentityRestoreV2:tensors:72*
_output_shapes
:*
T0�
AssignVariableOp_72AssignVariableOp#assignvariableop_72_conv11_1_kernelIdentity_72:output:0*
dtype0*
_output_shapes
 P
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp!assignvariableop_73_conv11_1_biasIdentity_73:output:0*
dtype0*
_output_shapes
 P
Identity_74IdentityRestoreV2:tensors:74*
_output_shapes
:*
T0�
AssignVariableOp_74AssignVariableOp"assignvariableop_74_norm11_1_gammaIdentity_74:output:0*
dtype0*
_output_shapes
 P
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp!assignvariableop_75_norm11_1_betaIdentity_75:output:0*
dtype0*
_output_shapes
 P
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp#assignvariableop_76_conv11_2_kernelIdentity_76:output:0*
_output_shapes
 *
dtype0P
Identity_77IdentityRestoreV2:tensors:77*
_output_shapes
:*
T0�
AssignVariableOp_77AssignVariableOp!assignvariableop_77_conv11_2_biasIdentity_77:output:0*
dtype0*
_output_shapes
 P
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp"assignvariableop_78_norm11_2_gammaIdentity_78:output:0*
dtype0*
_output_shapes
 P
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp!assignvariableop_79_norm11_2_betaIdentity_79:output:0*
dtype0*
_output_shapes
 P
Identity_80IdentityRestoreV2:tensors:80*
_output_shapes
:*
T0�
AssignVariableOp_80AssignVariableOp$assignvariableop_80_deconv1_1_kernelIdentity_80:output:0*
dtype0*
_output_shapes
 P
Identity_81IdentityRestoreV2:tensors:81*
_output_shapes
:*
T0�
AssignVariableOp_81AssignVariableOp"assignvariableop_81_deconv1_1_biasIdentity_81:output:0*
dtype0*
_output_shapes
 P
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp$assignvariableop_82_deconv1_2_kernelIdentity_82:output:0*
dtype0*
_output_shapes
 P
Identity_83IdentityRestoreV2:tensors:83*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp"assignvariableop_83_deconv1_2_biasIdentity_83:output:0*
dtype0*
_output_shapes
 P
Identity_84IdentityRestoreV2:tensors:84*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp&assignvariableop_84_norm_deconv1_gammaIdentity_84:output:0*
dtype0*
_output_shapes
 P
Identity_85IdentityRestoreV2:tensors:85*
_output_shapes
:*
T0�
AssignVariableOp_85AssignVariableOp%assignvariableop_85_norm_deconv1_betaIdentity_85:output:0*
dtype0*
_output_shapes
 P
Identity_86IdentityRestoreV2:tensors:86*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp$assignvariableop_86_deconv2_1_kernelIdentity_86:output:0*
dtype0*
_output_shapes
 P
Identity_87IdentityRestoreV2:tensors:87*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp"assignvariableop_87_deconv2_1_biasIdentity_87:output:0*
_output_shapes
 *
dtype0P
Identity_88IdentityRestoreV2:tensors:88*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp$assignvariableop_88_deconv2_2_kernelIdentity_88:output:0*
dtype0*
_output_shapes
 P
Identity_89IdentityRestoreV2:tensors:89*
_output_shapes
:*
T0�
AssignVariableOp_89AssignVariableOp"assignvariableop_89_deconv2_2_biasIdentity_89:output:0*
dtype0*
_output_shapes
 P
Identity_90IdentityRestoreV2:tensors:90*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp&assignvariableop_90_norm_deconv2_gammaIdentity_90:output:0*
dtype0*
_output_shapes
 P
Identity_91IdentityRestoreV2:tensors:91*
_output_shapes
:*
T0�
AssignVariableOp_91AssignVariableOp%assignvariableop_91_norm_deconv2_betaIdentity_91:output:0*
dtype0*
_output_shapes
 P
Identity_92IdentityRestoreV2:tensors:92*
_output_shapes
:*
T0�
AssignVariableOp_92AssignVariableOp"assignvariableop_92_deconv3_kernelIdentity_92:output:0*
dtype0*
_output_shapes
 P
Identity_93IdentityRestoreV2:tensors:93*
_output_shapes
:*
T0�
AssignVariableOp_93AssignVariableOp assignvariableop_93_deconv3_biasIdentity_93:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_94Identityfile_prefix^AssignVariableOp_27^AssignVariableOp_43^AssignVariableOp_49^AssignVariableOp_69^AssignVariableOp_76^AssignVariableOp_83^AssignVariableOp_91^AssignVariableOp_8^AssignVariableOp_15^AssignVariableOp_50^AssignVariableOp_62^AssignVariableOp_70^AssignVariableOp_77^AssignVariableOp_84^AssignVariableOp_9^AssignVariableOp_88^AssignVariableOp_51^AssignVariableOp_63^AssignVariableOp_71^AssignVariableOp_78^AssignVariableOp_85^AssignVariableOp_92^AssignVariableOp_52^AssignVariableOp_64^AssignVariableOp_72^AssignVariableOp_79^AssignVariableOp_86^AssignVariableOp_93^AssignVariableOp_65^AssignVariableOp_73^AssignVariableOp_80^AssignVariableOp_87^NoOp^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_58^AssignVariableOp_55^AssignVariableOp_59^AssignVariableOp_66^AssignVariableOp_56^AssignVariableOp_60^AssignVariableOp_67^AssignVariableOp_74^AssignVariableOp_57^AssignVariableOp_61^AssignVariableOp_68^AssignVariableOp_75^AssignVariableOp_81^AssignVariableOp_89^AssignVariableOp_4^AssignVariableOp_10^AssignVariableOp_16^AssignVariableOp_28^AssignVariableOp_35^AssignVariableOp_41^AssignVariableOp_48^AssignVariableOp_36^AssignVariableOp_42^AssignVariableOp_29^AssignVariableOp_44^AssignVariableOp_17^AssignVariableOp_22^AssignVariableOp_30^AssignVariableOp^AssignVariableOp_11^AssignVariableOp_19^AssignVariableOp_23^AssignVariableOp_31^AssignVariableOp_37^AssignVariableOp_3^AssignVariableOp_12^AssignVariableOp_20^AssignVariableOp_24^AssignVariableOp_32^AssignVariableOp_38^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_21^AssignVariableOp_25^AssignVariableOp_33^AssignVariableOp_39^AssignVariableOp_46^AssignVariableOp_1^AssignVariableOp_7^AssignVariableOp_18^AssignVariableOp_26^AssignVariableOp_34^AssignVariableOp_40^AssignVariableOp_47^AssignVariableOp_82^AssignVariableOp_90^AssignVariableOp_2^AssignVariableOp_13^AssignVariableOp_14"/device:CPU:0*
_output_shapes
: *
T0�
Identity_95IdentityIdentity_94:output:0^AssignVariableOp_27^AssignVariableOp_88^AssignVariableOp_20^AssignVariableOp_81^AssignVariableOp_13^AssignVariableOp_36^AssignVariableOp_59^AssignVariableOp_10^AssignVariableOp_90^AssignVariableOp_93^AssignVariableOp_39^AssignVariableOp_9^AssignVariableOp_32^AssignVariableOp_55^AssignVariableOp_78^AssignVariableOp_29^AssignVariableOp_18^AssignVariableOp_33^AssignVariableOp_28^AssignVariableOp_51^AssignVariableOp_74^AssignVariableOp_6^AssignVariableOp_48^AssignVariableOp_37^AssignVariableOp_63^AssignVariableOp_14^AssignVariableOp_75^AssignVariableOp_47^AssignVariableOp_70^AssignVariableOp_2^AssignVariableOp_25^AssignVariableOp_67^RestoreV2_1^AssignVariableOp_52^AssignVariableOp_82^AssignVariableOp_21^AssignVariableOp_44^AssignVariableOp_86^AssignVariableOp_56^AssignVariableOp_22^AssignVariableOp_7^AssignVariableOp_15^AssignVariableOp_57^AssignVariableOp_46^AssignVariableOp_16^AssignVariableOp_35^AssignVariableOp_58^AssignVariableOp_11^AssignVariableOp_38^AssignVariableOp_26^AssignVariableOp_30^AssignVariableOp_72^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_45^AssignVariableOp_49^AssignVariableOp^AssignVariableOp_42^AssignVariableOp_84^AssignVariableOp_54^AssignVariableOp_83^AssignVariableOp_34^AssignVariableOp_91^AssignVariableOp_61^AssignVariableOp_12^AssignVariableOp_73^AssignVariableOp_5^AssignVariableOp_41^AssignVariableOp_68^AssignVariableOp_19^AssignVariableOp_80^AssignVariableOp_31^AssignVariableOp_77^AssignVariableOp_24^AssignVariableOp_66^AssignVariableOp_60^AssignVariableOp_87^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_50^AssignVariableOp_1^AssignVariableOp_43
^RestoreV2^AssignVariableOp_89^AssignVariableOp_79^AssignVariableOp_53^AssignVariableOp_4^AssignVariableOp_65^AssignVariableOp_69^AssignVariableOp_92^AssignVariableOp_62^AssignVariableOp_85^AssignVariableOp_17^AssignVariableOp_40^AssignVariableOp_64^AssignVariableOp_71*
T0*
_output_shapes
: "#
identity_95Identity_95:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_79AssignVariableOp_792*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_89AssignVariableOp_892
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_59AssignVariableOp_59:> :? :@ :A :B :C :D :E :F :G :H :I :J :K :L :M :N :O :P :Q :R :S :T :U :V :W :X :Y :Z :[ :\ :] :^ :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := 
�J
� 
"__inference_signature_wrapper_7276	
input*
&statefulpartitionedcall_conv1_1_kernel(
$statefulpartitionedcall_conv1_1_bias'
#statefulpartitionedcall_norm1_gamma&
"statefulpartitionedcall_norm1_beta,
(statefulpartitionedcall_conv2_1_1_kernel*
&statefulpartitionedcall_conv2_1_1_bias,
(statefulpartitionedcall_conv2_2_1_kernel*
&statefulpartitionedcall_conv2_2_1_bias'
#statefulpartitionedcall_norm2_gamma&
"statefulpartitionedcall_norm2_beta,
(statefulpartitionedcall_conv3_1_1_kernel*
&statefulpartitionedcall_conv3_1_1_bias,
(statefulpartitionedcall_conv3_2_1_kernel*
&statefulpartitionedcall_conv3_2_1_bias'
#statefulpartitionedcall_norm3_gamma&
"statefulpartitionedcall_norm3_beta*
&statefulpartitionedcall_conv4_1_kernel(
$statefulpartitionedcall_conv4_1_bias)
%statefulpartitionedcall_norm4_1_gamma(
$statefulpartitionedcall_norm4_1_beta*
&statefulpartitionedcall_conv4_2_kernel(
$statefulpartitionedcall_conv4_2_bias)
%statefulpartitionedcall_norm4_2_gamma(
$statefulpartitionedcall_norm4_2_beta*
&statefulpartitionedcall_conv5_1_kernel(
$statefulpartitionedcall_conv5_1_bias)
%statefulpartitionedcall_norm5_1_gamma(
$statefulpartitionedcall_norm5_1_beta*
&statefulpartitionedcall_conv5_2_kernel(
$statefulpartitionedcall_conv5_2_bias)
%statefulpartitionedcall_norm5_2_gamma(
$statefulpartitionedcall_norm5_2_beta*
&statefulpartitionedcall_conv6_1_kernel(
$statefulpartitionedcall_conv6_1_bias)
%statefulpartitionedcall_norm6_1_gamma(
$statefulpartitionedcall_norm6_1_beta*
&statefulpartitionedcall_conv6_2_kernel(
$statefulpartitionedcall_conv6_2_bias)
%statefulpartitionedcall_norm6_2_gamma(
$statefulpartitionedcall_norm6_2_beta*
&statefulpartitionedcall_conv7_1_kernel(
$statefulpartitionedcall_conv7_1_bias)
%statefulpartitionedcall_norm7_1_gamma(
$statefulpartitionedcall_norm7_1_beta*
&statefulpartitionedcall_conv7_2_kernel(
$statefulpartitionedcall_conv7_2_bias)
%statefulpartitionedcall_norm7_2_gamma(
$statefulpartitionedcall_norm7_2_beta*
&statefulpartitionedcall_conv8_1_kernel(
$statefulpartitionedcall_conv8_1_bias)
%statefulpartitionedcall_norm8_1_gamma(
$statefulpartitionedcall_norm8_1_beta*
&statefulpartitionedcall_conv8_2_kernel(
$statefulpartitionedcall_conv8_2_bias)
%statefulpartitionedcall_norm8_2_gamma(
$statefulpartitionedcall_norm8_2_beta*
&statefulpartitionedcall_conv9_1_kernel(
$statefulpartitionedcall_conv9_1_bias)
%statefulpartitionedcall_norm9_1_gamma(
$statefulpartitionedcall_norm9_1_beta*
&statefulpartitionedcall_conv9_2_kernel(
$statefulpartitionedcall_conv9_2_bias)
%statefulpartitionedcall_norm9_2_gamma(
$statefulpartitionedcall_norm9_2_beta+
'statefulpartitionedcall_conv10_1_kernel)
%statefulpartitionedcall_conv10_1_bias*
&statefulpartitionedcall_norm10_1_gamma)
%statefulpartitionedcall_norm10_1_beta+
'statefulpartitionedcall_conv10_2_kernel)
%statefulpartitionedcall_conv10_2_bias*
&statefulpartitionedcall_norm10_2_gamma)
%statefulpartitionedcall_norm10_2_beta+
'statefulpartitionedcall_conv11_1_kernel)
%statefulpartitionedcall_conv11_1_bias*
&statefulpartitionedcall_norm11_1_gamma)
%statefulpartitionedcall_norm11_1_beta+
'statefulpartitionedcall_conv11_2_kernel)
%statefulpartitionedcall_conv11_2_bias*
&statefulpartitionedcall_norm11_2_gamma)
%statefulpartitionedcall_norm11_2_beta,
(statefulpartitionedcall_deconv1_1_kernel*
&statefulpartitionedcall_deconv1_1_bias,
(statefulpartitionedcall_deconv1_2_kernel*
&statefulpartitionedcall_deconv1_2_bias.
*statefulpartitionedcall_norm_deconv1_gamma-
)statefulpartitionedcall_norm_deconv1_beta,
(statefulpartitionedcall_deconv2_1_kernel*
&statefulpartitionedcall_deconv2_1_bias,
(statefulpartitionedcall_deconv2_2_kernel*
&statefulpartitionedcall_deconv2_2_bias.
*statefulpartitionedcall_norm_deconv2_gamma-
)statefulpartitionedcall_norm_deconv2_beta*
&statefulpartitionedcall_deconv3_kernel(
$statefulpartitionedcall_deconv3_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput&statefulpartitionedcall_conv1_1_kernel$statefulpartitionedcall_conv1_1_bias#statefulpartitionedcall_norm1_gamma"statefulpartitionedcall_norm1_beta(statefulpartitionedcall_conv2_1_1_kernel&statefulpartitionedcall_conv2_1_1_bias(statefulpartitionedcall_conv2_2_1_kernel&statefulpartitionedcall_conv2_2_1_bias#statefulpartitionedcall_norm2_gamma"statefulpartitionedcall_norm2_beta(statefulpartitionedcall_conv3_1_1_kernel&statefulpartitionedcall_conv3_1_1_bias(statefulpartitionedcall_conv3_2_1_kernel&statefulpartitionedcall_conv3_2_1_bias#statefulpartitionedcall_norm3_gamma"statefulpartitionedcall_norm3_beta&statefulpartitionedcall_conv4_1_kernel$statefulpartitionedcall_conv4_1_bias%statefulpartitionedcall_norm4_1_gamma$statefulpartitionedcall_norm4_1_beta&statefulpartitionedcall_conv4_2_kernel$statefulpartitionedcall_conv4_2_bias%statefulpartitionedcall_norm4_2_gamma$statefulpartitionedcall_norm4_2_beta&statefulpartitionedcall_conv5_1_kernel$statefulpartitionedcall_conv5_1_bias%statefulpartitionedcall_norm5_1_gamma$statefulpartitionedcall_norm5_1_beta&statefulpartitionedcall_conv5_2_kernel$statefulpartitionedcall_conv5_2_bias%statefulpartitionedcall_norm5_2_gamma$statefulpartitionedcall_norm5_2_beta&statefulpartitionedcall_conv6_1_kernel$statefulpartitionedcall_conv6_1_bias%statefulpartitionedcall_norm6_1_gamma$statefulpartitionedcall_norm6_1_beta&statefulpartitionedcall_conv6_2_kernel$statefulpartitionedcall_conv6_2_bias%statefulpartitionedcall_norm6_2_gamma$statefulpartitionedcall_norm6_2_beta&statefulpartitionedcall_conv7_1_kernel$statefulpartitionedcall_conv7_1_bias%statefulpartitionedcall_norm7_1_gamma$statefulpartitionedcall_norm7_1_beta&statefulpartitionedcall_conv7_2_kernel$statefulpartitionedcall_conv7_2_bias%statefulpartitionedcall_norm7_2_gamma$statefulpartitionedcall_norm7_2_beta&statefulpartitionedcall_conv8_1_kernel$statefulpartitionedcall_conv8_1_bias%statefulpartitionedcall_norm8_1_gamma$statefulpartitionedcall_norm8_1_beta&statefulpartitionedcall_conv8_2_kernel$statefulpartitionedcall_conv8_2_bias%statefulpartitionedcall_norm8_2_gamma$statefulpartitionedcall_norm8_2_beta&statefulpartitionedcall_conv9_1_kernel$statefulpartitionedcall_conv9_1_bias%statefulpartitionedcall_norm9_1_gamma$statefulpartitionedcall_norm9_1_beta&statefulpartitionedcall_conv9_2_kernel$statefulpartitionedcall_conv9_2_bias%statefulpartitionedcall_norm9_2_gamma$statefulpartitionedcall_norm9_2_beta'statefulpartitionedcall_conv10_1_kernel%statefulpartitionedcall_conv10_1_bias&statefulpartitionedcall_norm10_1_gamma%statefulpartitionedcall_norm10_1_beta'statefulpartitionedcall_conv10_2_kernel%statefulpartitionedcall_conv10_2_bias&statefulpartitionedcall_norm10_2_gamma%statefulpartitionedcall_norm10_2_beta'statefulpartitionedcall_conv11_1_kernel%statefulpartitionedcall_conv11_1_bias&statefulpartitionedcall_norm11_1_gamma%statefulpartitionedcall_norm11_1_beta'statefulpartitionedcall_conv11_2_kernel%statefulpartitionedcall_conv11_2_bias&statefulpartitionedcall_norm11_2_gamma%statefulpartitionedcall_norm11_2_beta(statefulpartitionedcall_deconv1_1_kernel&statefulpartitionedcall_deconv1_1_bias(statefulpartitionedcall_deconv1_2_kernel&statefulpartitionedcall_deconv1_2_bias*statefulpartitionedcall_norm_deconv1_gamma)statefulpartitionedcall_norm_deconv1_beta(statefulpartitionedcall_deconv2_1_kernel&statefulpartitionedcall_deconv2_1_bias(statefulpartitionedcall_deconv2_2_kernel&statefulpartitionedcall_deconv2_2_bias*statefulpartitionedcall_norm_deconv2_gamma)statefulpartitionedcall_norm_deconv2_beta&statefulpartitionedcall_deconv3_kernel$statefulpartitionedcall_deconv3_bias*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*j
Tinc
a2_*+
_gradient_op_typePartitionedCall-7179*(
f#R!
__inference__wrapped_model_7173�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*1
_output_shapes
:�����������*
T0"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :H :I :J :K :L :M :N :O :P :Q :R :S :T :U :V :W :X :Y :Z :[ :\ :] :^ :% !

_user_specified_nameinput: : : : : : : : :	 :
 : : : : : : : : : : : 
ۍ	
�Y
__inference__wrapped_model_7173	
input@
<cartoon_generator_conv1_conv2d_readvariableop_conv1_1_kernel?
;cartoon_generator_conv1_biasadd_readvariableop_conv1_1_bias>
:cartoon_generator_norm1_reshape_readvariableop_norm1_gamma?
;cartoon_generator_norm1_reshape_1_readvariableop_norm1_betaD
@cartoon_generator_conv2_1_conv2d_readvariableop_conv2_1_1_kernelC
?cartoon_generator_conv2_1_biasadd_readvariableop_conv2_1_1_biasD
@cartoon_generator_conv2_2_conv2d_readvariableop_conv2_2_1_kernelC
?cartoon_generator_conv2_2_biasadd_readvariableop_conv2_2_1_bias>
:cartoon_generator_norm2_reshape_readvariableop_norm2_gamma?
;cartoon_generator_norm2_reshape_1_readvariableop_norm2_betaD
@cartoon_generator_conv3_1_conv2d_readvariableop_conv3_1_1_kernelC
?cartoon_generator_conv3_1_biasadd_readvariableop_conv3_1_1_biasD
@cartoon_generator_conv3_2_conv2d_readvariableop_conv3_2_1_kernelC
?cartoon_generator_conv3_2_biasadd_readvariableop_conv3_2_1_bias>
:cartoon_generator_norm3_reshape_readvariableop_norm3_gamma?
;cartoon_generator_norm3_reshape_1_readvariableop_norm3_betaB
>cartoon_generator_conv4_1_conv2d_readvariableop_conv4_1_kernelA
=cartoon_generator_conv4_1_biasadd_readvariableop_conv4_1_biasB
>cartoon_generator_norm4_1_reshape_readvariableop_norm4_1_gammaC
?cartoon_generator_norm4_1_reshape_1_readvariableop_norm4_1_betaB
>cartoon_generator_conv4_2_conv2d_readvariableop_conv4_2_kernelA
=cartoon_generator_conv4_2_biasadd_readvariableop_conv4_2_biasB
>cartoon_generator_norm4_2_reshape_readvariableop_norm4_2_gammaC
?cartoon_generator_norm4_2_reshape_1_readvariableop_norm4_2_betaB
>cartoon_generator_conv5_1_conv2d_readvariableop_conv5_1_kernelA
=cartoon_generator_conv5_1_biasadd_readvariableop_conv5_1_biasB
>cartoon_generator_norm5_1_reshape_readvariableop_norm5_1_gammaC
?cartoon_generator_norm5_1_reshape_1_readvariableop_norm5_1_betaB
>cartoon_generator_conv5_2_conv2d_readvariableop_conv5_2_kernelA
=cartoon_generator_conv5_2_biasadd_readvariableop_conv5_2_biasB
>cartoon_generator_norm5_2_reshape_readvariableop_norm5_2_gammaC
?cartoon_generator_norm5_2_reshape_1_readvariableop_norm5_2_betaB
>cartoon_generator_conv6_1_conv2d_readvariableop_conv6_1_kernelA
=cartoon_generator_conv6_1_biasadd_readvariableop_conv6_1_biasB
>cartoon_generator_norm6_1_reshape_readvariableop_norm6_1_gammaC
?cartoon_generator_norm6_1_reshape_1_readvariableop_norm6_1_betaB
>cartoon_generator_conv6_2_conv2d_readvariableop_conv6_2_kernelA
=cartoon_generator_conv6_2_biasadd_readvariableop_conv6_2_biasB
>cartoon_generator_norm6_2_reshape_readvariableop_norm6_2_gammaC
?cartoon_generator_norm6_2_reshape_1_readvariableop_norm6_2_betaB
>cartoon_generator_conv7_1_conv2d_readvariableop_conv7_1_kernelA
=cartoon_generator_conv7_1_biasadd_readvariableop_conv7_1_biasB
>cartoon_generator_norm7_1_reshape_readvariableop_norm7_1_gammaC
?cartoon_generator_norm7_1_reshape_1_readvariableop_norm7_1_betaB
>cartoon_generator_conv7_2_conv2d_readvariableop_conv7_2_kernelA
=cartoon_generator_conv7_2_biasadd_readvariableop_conv7_2_biasB
>cartoon_generator_norm7_2_reshape_readvariableop_norm7_2_gammaC
?cartoon_generator_norm7_2_reshape_1_readvariableop_norm7_2_betaB
>cartoon_generator_conv8_1_conv2d_readvariableop_conv8_1_kernelA
=cartoon_generator_conv8_1_biasadd_readvariableop_conv8_1_biasB
>cartoon_generator_norm8_1_reshape_readvariableop_norm8_1_gammaC
?cartoon_generator_norm8_1_reshape_1_readvariableop_norm8_1_betaB
>cartoon_generator_conv8_2_conv2d_readvariableop_conv8_2_kernelA
=cartoon_generator_conv8_2_biasadd_readvariableop_conv8_2_biasB
>cartoon_generator_norm8_2_reshape_readvariableop_norm8_2_gammaC
?cartoon_generator_norm8_2_reshape_1_readvariableop_norm8_2_betaB
>cartoon_generator_conv9_1_conv2d_readvariableop_conv9_1_kernelA
=cartoon_generator_conv9_1_biasadd_readvariableop_conv9_1_biasB
>cartoon_generator_norm9_1_reshape_readvariableop_norm9_1_gammaC
?cartoon_generator_norm9_1_reshape_1_readvariableop_norm9_1_betaB
>cartoon_generator_conv9_2_conv2d_readvariableop_conv9_2_kernelA
=cartoon_generator_conv9_2_biasadd_readvariableop_conv9_2_biasB
>cartoon_generator_norm9_2_reshape_readvariableop_norm9_2_gammaC
?cartoon_generator_norm9_2_reshape_1_readvariableop_norm9_2_betaD
@cartoon_generator_conv10_1_conv2d_readvariableop_conv10_1_kernelC
?cartoon_generator_conv10_1_biasadd_readvariableop_conv10_1_biasD
@cartoon_generator_norm10_1_reshape_readvariableop_norm10_1_gammaE
Acartoon_generator_norm10_1_reshape_1_readvariableop_norm10_1_betaD
@cartoon_generator_conv10_2_conv2d_readvariableop_conv10_2_kernelC
?cartoon_generator_conv10_2_biasadd_readvariableop_conv10_2_biasD
@cartoon_generator_norm10_2_reshape_readvariableop_norm10_2_gammaE
Acartoon_generator_norm10_2_reshape_1_readvariableop_norm10_2_betaD
@cartoon_generator_conv11_1_conv2d_readvariableop_conv11_1_kernelC
?cartoon_generator_conv11_1_biasadd_readvariableop_conv11_1_biasD
@cartoon_generator_norm11_1_reshape_readvariableop_norm11_1_gammaE
Acartoon_generator_norm11_1_reshape_1_readvariableop_norm11_1_betaD
@cartoon_generator_conv11_2_conv2d_readvariableop_conv11_2_kernelC
?cartoon_generator_conv11_2_biasadd_readvariableop_conv11_2_biasD
@cartoon_generator_norm11_2_reshape_readvariableop_norm11_2_gammaE
Acartoon_generator_norm11_2_reshape_1_readvariableop_norm11_2_betaP
Lcartoon_generator_deconv1_1_conv2d_transpose_readvariableop_deconv1_1_kernelE
Acartoon_generator_deconv1_1_biasadd_readvariableop_deconv1_1_biasF
Bcartoon_generator_deconv1_2_conv2d_readvariableop_deconv1_2_kernelE
Acartoon_generator_deconv1_2_biasadd_readvariableop_deconv1_2_biasL
Hcartoon_generator_norm_deconv1_reshape_readvariableop_norm_deconv1_gammaM
Icartoon_generator_norm_deconv1_reshape_1_readvariableop_norm_deconv1_betaP
Lcartoon_generator_deconv2_1_conv2d_transpose_readvariableop_deconv2_1_kernelE
Acartoon_generator_deconv2_1_biasadd_readvariableop_deconv2_1_biasF
Bcartoon_generator_deconv2_2_conv2d_readvariableop_deconv2_2_kernelE
Acartoon_generator_deconv2_2_biasadd_readvariableop_deconv2_2_biasL
Hcartoon_generator_norm_deconv2_reshape_readvariableop_norm_deconv2_gammaM
Icartoon_generator_norm_deconv2_reshape_1_readvariableop_norm_deconv2_betaB
>cartoon_generator_deconv3_conv2d_readvariableop_deconv3_kernelA
=cartoon_generator_deconv3_biasadd_readvariableop_deconv3_bias
identity��0Cartoon_Generator/norm6_1/Reshape/ReadVariableOp�0Cartoon_Generator/conv9_2/BiasAdd/ReadVariableOp�0Cartoon_Generator/conv11_1/Conv2D/ReadVariableOp�2Cartoon_Generator/deconv1_2/BiasAdd/ReadVariableOp�3Cartoon_Generator/norm11_2/Reshape_1/ReadVariableOp�/Cartoon_Generator/conv5_2/Conv2D/ReadVariableOp�/Cartoon_Generator/conv4_2/Conv2D/ReadVariableOp�0Cartoon_Generator/conv8_1/BiasAdd/ReadVariableOp�0Cartoon_Generator/norm5_1/Reshape/ReadVariableOp�/Cartoon_Generator/conv9_2/Conv2D/ReadVariableOp�1Cartoon_Generator/norm10_2/Reshape/ReadVariableOp�3Cartoon_Generator/norm10_1/Reshape_1/ReadVariableOp�1Cartoon_Generator/deconv1_2/Conv2D/ReadVariableOp�0Cartoon_Generator/conv3_1/BiasAdd/ReadVariableOp�/Cartoon_Generator/conv8_1/Conv2D/ReadVariableOp�0Cartoon_Generator/conv6_2/BiasAdd/ReadVariableOp�2Cartoon_Generator/norm8_2/Reshape_1/ReadVariableOp�0Cartoon_Generator/norm9_1/Reshape/ReadVariableOp�/Cartoon_Generator/conv6_1/Conv2D/ReadVariableOp�0Cartoon_Generator/norm4_1/Reshape/ReadVariableOp�1Cartoon_Generator/conv11_2/BiasAdd/ReadVariableOp�0Cartoon_Generator/deconv3/BiasAdd/ReadVariableOp�/Cartoon_Generator/conv6_2/Conv2D/ReadVariableOp�0Cartoon_Generator/conv4_2/BiasAdd/ReadVariableOp�2Cartoon_Generator/norm4_2/Reshape_1/ReadVariableOp�2Cartoon_Generator/norm7_1/Reshape_1/ReadVariableOp�0Cartoon_Generator/norm7_2/Reshape/ReadVariableOp�0Cartoon_Generator/conv11_2/Conv2D/ReadVariableOp�/Cartoon_Generator/deconv3/Conv2D/ReadVariableOp�1Cartoon_Generator/conv10_1/BiasAdd/ReadVariableOp�2Cartoon_Generator/norm6_2/Reshape_1/ReadVariableOp�0Cartoon_Generator/conv10_1/Conv2D/ReadVariableOp�0Cartoon_Generator/conv8_2/BiasAdd/ReadVariableOp�0Cartoon_Generator/norm5_2/Reshape/ReadVariableOp�/Cartoon_Generator/conv2_2/Conv2D/ReadVariableOp�2Cartoon_Generator/norm5_1/Reshape_1/ReadVariableOp�0Cartoon_Generator/conv5_2/BiasAdd/ReadVariableOp�.Cartoon_Generator/conv1/BiasAdd/ReadVariableOp�3Cartoon_Generator/norm10_2/Reshape_1/ReadVariableOp�1Cartoon_Generator/norm11_1/Reshape/ReadVariableOp�5Cartoon_Generator/norm_deconv2/Reshape/ReadVariableOp�0Cartoon_Generator/conv6_1/BiasAdd/ReadVariableOp�/Cartoon_Generator/conv8_2/Conv2D/ReadVariableOp�0Cartoon_Generator/norm3/Reshape_1/ReadVariableOp�.Cartoon_Generator/norm1/Reshape/ReadVariableOp�0Cartoon_Generator/conv7_1/BiasAdd/ReadVariableOp�0Cartoon_Generator/conv4_1/BiasAdd/ReadVariableOp�2Cartoon_Generator/norm9_1/Reshape_1/ReadVariableOp�0Cartoon_Generator/norm9_2/Reshape/ReadVariableOp�/Cartoon_Generator/conv5_1/Conv2D/ReadVariableOp�/Cartoon_Generator/conv4_1/Conv2D/ReadVariableOp�/Cartoon_Generator/conv7_1/Conv2D/ReadVariableOp�2Cartoon_Generator/deconv2_1/BiasAdd/ReadVariableOp�2Cartoon_Generator/norm4_1/Reshape_1/ReadVariableOp�0Cartoon_Generator/norm1/Reshape_1/ReadVariableOp�/Cartoon_Generator/conv2_1/Conv2D/ReadVariableOp�5Cartoon_Generator/norm_deconv1/Reshape/ReadVariableOp�2Cartoon_Generator/norm7_2/Reshape_1/ReadVariableOp�0Cartoon_Generator/norm8_1/Reshape/ReadVariableOp�1Cartoon_Generator/conv10_2/BiasAdd/ReadVariableOp�0Cartoon_Generator/conv2_2/BiasAdd/ReadVariableOp�2Cartoon_Generator/norm6_1/Reshape_1/ReadVariableOp�;Cartoon_Generator/deconv2_1/conv2d_transpose/ReadVariableOp�0Cartoon_Generator/norm6_2/Reshape/ReadVariableOp�0Cartoon_Generator/conv10_2/Conv2D/ReadVariableOp�0Cartoon_Generator/conv9_1/BiasAdd/ReadVariableOp�2Cartoon_Generator/deconv1_1/BiasAdd/ReadVariableOp�-Cartoon_Generator/conv1/Conv2D/ReadVariableOp�3Cartoon_Generator/norm11_1/Reshape_1/ReadVariableOp�1Cartoon_Generator/norm11_2/Reshape/ReadVariableOp�.Cartoon_Generator/norm2/Reshape/ReadVariableOp�7Cartoon_Generator/norm_deconv2/Reshape_1/ReadVariableOp�/Cartoon_Generator/conv3_2/Conv2D/ReadVariableOp�0Cartoon_Generator/norm4_2/Reshape/ReadVariableOp�.Cartoon_Generator/norm3/Reshape/ReadVariableOp�/Cartoon_Generator/conv9_1/Conv2D/ReadVariableOp�1Cartoon_Generator/deconv2_2/Conv2D/ReadVariableOp�0Cartoon_Generator/conv7_2/BiasAdd/ReadVariableOp�1Cartoon_Generator/norm10_1/Reshape/ReadVariableOp�2Cartoon_Generator/norm9_2/Reshape_1/ReadVariableOp�;Cartoon_Generator/deconv1_1/conv2d_transpose/ReadVariableOp�/Cartoon_Generator/conv7_2/Conv2D/ReadVariableOp�0Cartoon_Generator/norm2/Reshape_1/ReadVariableOp�/Cartoon_Generator/conv3_1/Conv2D/ReadVariableOp�7Cartoon_Generator/norm_deconv1/Reshape_1/ReadVariableOp�0Cartoon_Generator/norm8_2/Reshape/ReadVariableOp�0Cartoon_Generator/conv5_1/BiasAdd/ReadVariableOp�2Cartoon_Generator/norm8_1/Reshape_1/ReadVariableOp�0Cartoon_Generator/conv3_2/BiasAdd/ReadVariableOp�2Cartoon_Generator/deconv2_2/BiasAdd/ReadVariableOp�0Cartoon_Generator/conv2_1/BiasAdd/ReadVariableOp�1Cartoon_Generator/conv11_1/BiasAdd/ReadVariableOp�2Cartoon_Generator/norm5_2/Reshape_1/ReadVariableOp�0Cartoon_Generator/norm7_1/Reshape/ReadVariableOp�
-Cartoon_Generator/zero_padding2d/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
$Cartoon_Generator/zero_padding2d/PadPadinput6Cartoon_Generator/zero_padding2d/Pad/paddings:output:0*
T0*1
_output_shapes
:������������
-Cartoon_Generator/conv1/Conv2D/ReadVariableOpReadVariableOp<cartoon_generator_conv1_conv2d_readvariableop_conv1_1_kernel*
dtype0*&
_output_shapes
:@�
Cartoon_Generator/conv1/Conv2DConv2D-Cartoon_Generator/zero_padding2d/Pad:output:05Cartoon_Generator/conv1/Conv2D/ReadVariableOp:value:0*
paddingVALID*1
_output_shapes
:�����������@*
T0*
strides
�
.Cartoon_Generator/conv1/BiasAdd/ReadVariableOpReadVariableOp;cartoon_generator_conv1_biasadd_readvariableop_conv1_1_bias*
dtype0*
_output_shapes
:@�
Cartoon_Generator/conv1/BiasAddBiasAdd'Cartoon_Generator/conv1/Conv2D:output:06Cartoon_Generator/conv1/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:�����������@*
T0�
.Cartoon_Generator/norm1/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm1/MeanMean(Cartoon_Generator/conv1/BiasAdd:output:07Cartoon_Generator/norm1/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
ICartoon_Generator/norm1/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
7Cartoon_Generator/norm1/reduce_std/reduce_variance/MeanMean(Cartoon_Generator/conv1/BiasAdd:output:0RCartoon_Generator/norm1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
6Cartoon_Generator/norm1/reduce_std/reduce_variance/subSub(Cartoon_Generator/conv1/BiasAdd:output:0@Cartoon_Generator/norm1/reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:�����������@�
9Cartoon_Generator/norm1/reduce_std/reduce_variance/SquareSquare:Cartoon_Generator/norm1/reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:�����������@�
KCartoon_Generator/norm1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         �
9Cartoon_Generator/norm1/reduce_std/reduce_variance/Mean_1Mean=Cartoon_Generator/norm1/reduce_std/reduce_variance/Square:y:0TCartoon_Generator/norm1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
'Cartoon_Generator/norm1/reduce_std/SqrtSqrtBCartoon_Generator/norm1/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������b
Cartoon_Generator/norm1/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm1/addAdd+Cartoon_Generator/norm1/reduce_std/Sqrt:y:0&Cartoon_Generator/norm1/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm1/subSub(Cartoon_Generator/conv1/BiasAdd:output:0%Cartoon_Generator/norm1/Mean:output:0*1
_output_shapes
:�����������@*
T0�
Cartoon_Generator/norm1/truedivRealDivCartoon_Generator/norm1/sub:z:0Cartoon_Generator/norm1/add:z:0*
T0*1
_output_shapes
:�����������@�
.Cartoon_Generator/norm1/Reshape/ReadVariableOpReadVariableOp:cartoon_generator_norm1_reshape_readvariableop_norm1_gamma*
dtype0*
_output_shapes
:~
%Cartoon_Generator/norm1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm1/ReshapeReshape6Cartoon_Generator/norm1/Reshape/ReadVariableOp:value:0.Cartoon_Generator/norm1/Reshape/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm1/mulMul#Cartoon_Generator/norm1/truediv:z:0(Cartoon_Generator/norm1/Reshape:output:0*
T0*1
_output_shapes
:�����������@�
0Cartoon_Generator/norm1/Reshape_1/ReadVariableOpReadVariableOp;cartoon_generator_norm1_reshape_1_readvariableop_norm1_beta*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm1/Reshape_1/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0�
!Cartoon_Generator/norm1/Reshape_1Reshape8Cartoon_Generator/norm1/Reshape_1/ReadVariableOp:value:00Cartoon_Generator/norm1/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm1/add_1AddCartoon_Generator/norm1/mul:z:0*Cartoon_Generator/norm1/Reshape_1:output:0*
T0*1
_output_shapes
:�����������@�
!Cartoon_Generator/activation/ReluRelu!Cartoon_Generator/norm1/add_1:z:0*1
_output_shapes
:�����������@*
T0�
/Cartoon_Generator/conv2_1/Conv2D/ReadVariableOpReadVariableOp@cartoon_generator_conv2_1_conv2d_readvariableop_conv2_1_1_kernel*'
_output_shapes
:@�*
dtype0�
 Cartoon_Generator/conv2_1/Conv2DConv2D/Cartoon_Generator/activation/Relu:activations:07Cartoon_Generator/conv2_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*2
_output_shapes 
:�������������
0Cartoon_Generator/conv2_1/BiasAdd/ReadVariableOpReadVariableOp?cartoon_generator_conv2_1_biasadd_readvariableop_conv2_1_1_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv2_1/BiasAddBiasAdd)Cartoon_Generator/conv2_1/Conv2D:output:08Cartoon_Generator/conv2_1/BiasAdd/ReadVariableOp:value:0*2
_output_shapes 
:������������*
T0�
/Cartoon_Generator/conv2_2/Conv2D/ReadVariableOpReadVariableOp@cartoon_generator_conv2_2_conv2d_readvariableop_conv2_2_1_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv2_2/Conv2DConv2D*Cartoon_Generator/conv2_1/BiasAdd:output:07Cartoon_Generator/conv2_2/Conv2D/ReadVariableOp:value:0*
paddingSAME*2
_output_shapes 
:������������*
T0*
strides
�
0Cartoon_Generator/conv2_2/BiasAdd/ReadVariableOpReadVariableOp?cartoon_generator_conv2_2_biasadd_readvariableop_conv2_2_1_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv2_2/BiasAddBiasAdd)Cartoon_Generator/conv2_2/Conv2D:output:08Cartoon_Generator/conv2_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:�������������
.Cartoon_Generator/norm2/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm2/MeanMean*Cartoon_Generator/conv2_2/BiasAdd:output:07Cartoon_Generator/norm2/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
ICartoon_Generator/norm2/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
7Cartoon_Generator/norm2/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv2_2/BiasAdd:output:0RCartoon_Generator/norm2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
6Cartoon_Generator/norm2/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv2_2/BiasAdd:output:0@Cartoon_Generator/norm2/reduce_std/reduce_variance/Mean:output:0*2
_output_shapes 
:������������*
T0�
9Cartoon_Generator/norm2/reduce_std/reduce_variance/SquareSquare:Cartoon_Generator/norm2/reduce_std/reduce_variance/sub:z:0*2
_output_shapes 
:������������*
T0�
KCartoon_Generator/norm2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0�
9Cartoon_Generator/norm2/reduce_std/reduce_variance/Mean_1Mean=Cartoon_Generator/norm2/reduce_std/reduce_variance/Square:y:0TCartoon_Generator/norm2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
'Cartoon_Generator/norm2/reduce_std/SqrtSqrtBCartoon_Generator/norm2/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������b
Cartoon_Generator/norm2/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:�
Cartoon_Generator/norm2/addAdd+Cartoon_Generator/norm2/reduce_std/Sqrt:y:0&Cartoon_Generator/norm2/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm2/subSub*Cartoon_Generator/conv2_2/BiasAdd:output:0%Cartoon_Generator/norm2/Mean:output:0*
T0*2
_output_shapes 
:�������������
Cartoon_Generator/norm2/truedivRealDivCartoon_Generator/norm2/sub:z:0Cartoon_Generator/norm2/add:z:0*2
_output_shapes 
:������������*
T0�
.Cartoon_Generator/norm2/Reshape/ReadVariableOpReadVariableOp:cartoon_generator_norm2_reshape_readvariableop_norm2_gamma*
dtype0*
_output_shapes
:~
%Cartoon_Generator/norm2/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm2/ReshapeReshape6Cartoon_Generator/norm2/Reshape/ReadVariableOp:value:0.Cartoon_Generator/norm2/Reshape/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm2/mulMul#Cartoon_Generator/norm2/truediv:z:0(Cartoon_Generator/norm2/Reshape:output:0*
T0*2
_output_shapes 
:�������������
0Cartoon_Generator/norm2/Reshape_1/ReadVariableOpReadVariableOp;cartoon_generator_norm2_reshape_1_readvariableop_norm2_beta*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm2/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm2/Reshape_1Reshape8Cartoon_Generator/norm2/Reshape_1/ReadVariableOp:value:00Cartoon_Generator/norm2/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm2/add_1AddCartoon_Generator/norm2/mul:z:0*Cartoon_Generator/norm2/Reshape_1:output:0*2
_output_shapes 
:������������*
T0�
#Cartoon_Generator/activation_1/ReluRelu!Cartoon_Generator/norm2/add_1:z:0*
T0*2
_output_shapes 
:�������������
/Cartoon_Generator/conv3_1/Conv2D/ReadVariableOpReadVariableOp@cartoon_generator_conv3_1_conv2d_readvariableop_conv3_1_1_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv3_1/Conv2DConv2D1Cartoon_Generator/activation_1/Relu:activations:07Cartoon_Generator/conv3_1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/conv3_1/BiasAdd/ReadVariableOpReadVariableOp?cartoon_generator_conv3_1_biasadd_readvariableop_conv3_1_1_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv3_1/BiasAddBiasAdd)Cartoon_Generator/conv3_1/Conv2D:output:08Cartoon_Generator/conv3_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0�
/Cartoon_Generator/conv3_2/Conv2D/ReadVariableOpReadVariableOp@cartoon_generator_conv3_2_conv2d_readvariableop_conv3_2_1_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv3_2/Conv2DConv2D*Cartoon_Generator/conv3_1/BiasAdd:output:07Cartoon_Generator/conv3_2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0*
strides
*
paddingSAME�
0Cartoon_Generator/conv3_2/BiasAdd/ReadVariableOpReadVariableOp?cartoon_generator_conv3_2_biasadd_readvariableop_conv3_2_1_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv3_2/BiasAddBiasAdd)Cartoon_Generator/conv3_2/Conv2D:output:08Cartoon_Generator/conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
.Cartoon_Generator/norm3/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm3/MeanMean*Cartoon_Generator/conv3_2/BiasAdd:output:07Cartoon_Generator/norm3/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
ICartoon_Generator/norm3/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
7Cartoon_Generator/norm3/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv3_2/BiasAdd:output:0RCartoon_Generator/norm3/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
6Cartoon_Generator/norm3/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv3_2/BiasAdd:output:0@Cartoon_Generator/norm3/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
9Cartoon_Generator/norm3/reduce_std/reduce_variance/SquareSquare:Cartoon_Generator/norm3/reduce_std/reduce_variance/sub:z:0*0
_output_shapes
:���������@@�*
T0�
KCartoon_Generator/norm3/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm3/reduce_std/reduce_variance/Mean_1Mean=Cartoon_Generator/norm3/reduce_std/reduce_variance/Square:y:0TCartoon_Generator/norm3/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
'Cartoon_Generator/norm3/reduce_std/SqrtSqrtBCartoon_Generator/norm3/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������b
Cartoon_Generator/norm3/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm3/addAdd+Cartoon_Generator/norm3/reduce_std/Sqrt:y:0&Cartoon_Generator/norm3/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm3/subSub*Cartoon_Generator/conv3_2/BiasAdd:output:0%Cartoon_Generator/norm3/Mean:output:0*
T0*0
_output_shapes
:���������@@��
Cartoon_Generator/norm3/truedivRealDivCartoon_Generator/norm3/sub:z:0Cartoon_Generator/norm3/add:z:0*0
_output_shapes
:���������@@�*
T0�
.Cartoon_Generator/norm3/Reshape/ReadVariableOpReadVariableOp:cartoon_generator_norm3_reshape_readvariableop_norm3_gamma*
dtype0*
_output_shapes
:~
%Cartoon_Generator/norm3/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            �
Cartoon_Generator/norm3/ReshapeReshape6Cartoon_Generator/norm3/Reshape/ReadVariableOp:value:0.Cartoon_Generator/norm3/Reshape/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm3/mulMul#Cartoon_Generator/norm3/truediv:z:0(Cartoon_Generator/norm3/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm3/Reshape_1/ReadVariableOpReadVariableOp;cartoon_generator_norm3_reshape_1_readvariableop_norm3_beta*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm3/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm3/Reshape_1Reshape8Cartoon_Generator/norm3/Reshape_1/ReadVariableOp:value:00Cartoon_Generator/norm3/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm3/add_1AddCartoon_Generator/norm3/mul:z:0*Cartoon_Generator/norm3/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
#Cartoon_Generator/activation_2/ReluRelu!Cartoon_Generator/norm3/add_1:z:0*0
_output_shapes
:���������@@�*
T0�
/Cartoon_Generator/zero_padding2d_1/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
&Cartoon_Generator/zero_padding2d_1/PadPad1Cartoon_Generator/activation_2/Relu:activations:08Cartoon_Generator/zero_padding2d_1/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
/Cartoon_Generator/conv4_1/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv4_1_conv2d_readvariableop_conv4_1_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv4_1/Conv2DConv2D/Cartoon_Generator/zero_padding2d_1/Pad:output:07Cartoon_Generator/conv4_1/Conv2D/ReadVariableOp:value:0*
paddingVALID*0
_output_shapes
:���������@@�*
T0*
strides
�
0Cartoon_Generator/conv4_1/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv4_1_biasadd_readvariableop_conv4_1_bias*
_output_shapes	
:�*
dtype0�
!Cartoon_Generator/conv4_1/BiasAddBiasAdd)Cartoon_Generator/conv4_1/Conv2D:output:08Cartoon_Generator/conv4_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm4_1/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         �
Cartoon_Generator/norm4_1/MeanMean*Cartoon_Generator/conv4_1/BiasAdd:output:09Cartoon_Generator/norm4_1/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
KCartoon_Generator/norm4_1/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm4_1/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv4_1/BiasAdd:output:0TCartoon_Generator/norm4_1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
8Cartoon_Generator/norm4_1/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv4_1/BiasAdd:output:0BCartoon_Generator/norm4_1/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
;Cartoon_Generator/norm4_1/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm4_1/reduce_std/reduce_variance/sub:z:0*0
_output_shapes
:���������@@�*
T0�
MCartoon_Generator/norm4_1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm4_1/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm4_1/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm4_1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
)Cartoon_Generator/norm4_1/reduce_std/SqrtSqrtDCartoon_Generator/norm4_1/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm4_1/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm4_1/addAdd-Cartoon_Generator/norm4_1/reduce_std/Sqrt:y:0(Cartoon_Generator/norm4_1/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm4_1/subSub*Cartoon_Generator/conv4_1/BiasAdd:output:0'Cartoon_Generator/norm4_1/Mean:output:0*
T0*0
_output_shapes
:���������@@��
!Cartoon_Generator/norm4_1/truedivRealDiv!Cartoon_Generator/norm4_1/sub:z:0!Cartoon_Generator/norm4_1/add:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm4_1/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm4_1_reshape_readvariableop_norm4_1_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm4_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm4_1/ReshapeReshape8Cartoon_Generator/norm4_1/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm4_1/Reshape/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm4_1/mulMul%Cartoon_Generator/norm4_1/truediv:z:0*Cartoon_Generator/norm4_1/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm4_1/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm4_1_reshape_1_readvariableop_norm4_1_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm4_1/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm4_1/Reshape_1Reshape:Cartoon_Generator/norm4_1/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm4_1/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm4_1/add_1Add!Cartoon_Generator/norm4_1/mul:z:0,Cartoon_Generator/norm4_1/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
#Cartoon_Generator/activation_3/ReluRelu#Cartoon_Generator/norm4_1/add_1:z:0*
T0*0
_output_shapes
:���������@@��
/Cartoon_Generator/zero_padding2d_2/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
&Cartoon_Generator/zero_padding2d_2/PadPad1Cartoon_Generator/activation_3/Relu:activations:08Cartoon_Generator/zero_padding2d_2/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
/Cartoon_Generator/conv4_2/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv4_2_conv2d_readvariableop_conv4_2_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv4_2/Conv2DConv2D/Cartoon_Generator/zero_padding2d_2/Pad:output:07Cartoon_Generator/conv4_2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0*
strides
*
paddingVALID�
0Cartoon_Generator/conv4_2/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv4_2_biasadd_readvariableop_conv4_2_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv4_2/BiasAddBiasAdd)Cartoon_Generator/conv4_2/Conv2D:output:08Cartoon_Generator/conv4_2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm4_2/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm4_2/MeanMean*Cartoon_Generator/conv4_2/BiasAdd:output:09Cartoon_Generator/norm4_2/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
KCartoon_Generator/norm4_2/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm4_2/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv4_2/BiasAdd:output:0TCartoon_Generator/norm4_2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
8Cartoon_Generator/norm4_2/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv4_2/BiasAdd:output:0BCartoon_Generator/norm4_2/reduce_std/reduce_variance/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
;Cartoon_Generator/norm4_2/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm4_2/reduce_std/reduce_variance/sub:z:0*0
_output_shapes
:���������@@�*
T0�
MCartoon_Generator/norm4_2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm4_2/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm4_2/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm4_2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
)Cartoon_Generator/norm4_2/reduce_std/SqrtSqrtDCartoon_Generator/norm4_2/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm4_2/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm4_2/addAdd-Cartoon_Generator/norm4_2/reduce_std/Sqrt:y:0(Cartoon_Generator/norm4_2/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm4_2/subSub*Cartoon_Generator/conv4_2/BiasAdd:output:0'Cartoon_Generator/norm4_2/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
!Cartoon_Generator/norm4_2/truedivRealDiv!Cartoon_Generator/norm4_2/sub:z:0!Cartoon_Generator/norm4_2/add:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm4_2/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm4_2_reshape_readvariableop_norm4_2_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm4_2/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm4_2/ReshapeReshape8Cartoon_Generator/norm4_2/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm4_2/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm4_2/mulMul%Cartoon_Generator/norm4_2/truediv:z:0*Cartoon_Generator/norm4_2/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm4_2/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm4_2_reshape_1_readvariableop_norm4_2_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm4_2/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            �
#Cartoon_Generator/norm4_2/Reshape_1Reshape:Cartoon_Generator/norm4_2/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm4_2/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm4_2/add_1Add!Cartoon_Generator/norm4_2/mul:z:0,Cartoon_Generator/norm4_2/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
Cartoon_Generator/add/addAdd#Cartoon_Generator/norm4_2/add_1:z:01Cartoon_Generator/activation_2/Relu:activations:0*
T0*0
_output_shapes
:���������@@��
/Cartoon_Generator/zero_padding2d_3/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
&Cartoon_Generator/zero_padding2d_3/PadPadCartoon_Generator/add/add:z:08Cartoon_Generator/zero_padding2d_3/Pad/paddings:output:0*0
_output_shapes
:���������BB�*
T0�
/Cartoon_Generator/conv5_1/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv5_1_conv2d_readvariableop_conv5_1_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv5_1/Conv2DConv2D/Cartoon_Generator/zero_padding2d_3/Pad:output:07Cartoon_Generator/conv5_1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0*
strides
*
paddingVALID�
0Cartoon_Generator/conv5_1/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv5_1_biasadd_readvariableop_conv5_1_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv5_1/BiasAddBiasAdd)Cartoon_Generator/conv5_1/Conv2D:output:08Cartoon_Generator/conv5_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm5_1/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm5_1/MeanMean*Cartoon_Generator/conv5_1/BiasAdd:output:09Cartoon_Generator/norm5_1/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
KCartoon_Generator/norm5_1/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm5_1/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv5_1/BiasAdd:output:0TCartoon_Generator/norm5_1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
8Cartoon_Generator/norm5_1/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv5_1/BiasAdd:output:0BCartoon_Generator/norm5_1/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
;Cartoon_Generator/norm5_1/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm5_1/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
MCartoon_Generator/norm5_1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm5_1/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm5_1/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm5_1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
)Cartoon_Generator/norm5_1/reduce_std/SqrtSqrtDCartoon_Generator/norm5_1/reduce_std/reduce_variance/Mean_1:output:0*/
_output_shapes
:���������*
T0d
Cartoon_Generator/norm5_1/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm5_1/addAdd-Cartoon_Generator/norm5_1/reduce_std/Sqrt:y:0(Cartoon_Generator/norm5_1/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm5_1/subSub*Cartoon_Generator/conv5_1/BiasAdd:output:0'Cartoon_Generator/norm5_1/Mean:output:0*
T0*0
_output_shapes
:���������@@��
!Cartoon_Generator/norm5_1/truedivRealDiv!Cartoon_Generator/norm5_1/sub:z:0!Cartoon_Generator/norm5_1/add:z:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm5_1/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm5_1_reshape_readvariableop_norm5_1_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm5_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm5_1/ReshapeReshape8Cartoon_Generator/norm5_1/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm5_1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm5_1/mulMul%Cartoon_Generator/norm5_1/truediv:z:0*Cartoon_Generator/norm5_1/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm5_1/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm5_1_reshape_1_readvariableop_norm5_1_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm5_1/Reshape_1/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0�
#Cartoon_Generator/norm5_1/Reshape_1Reshape:Cartoon_Generator/norm5_1/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm5_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm5_1/add_1Add!Cartoon_Generator/norm5_1/mul:z:0,Cartoon_Generator/norm5_1/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
#Cartoon_Generator/activation_4/ReluRelu#Cartoon_Generator/norm5_1/add_1:z:0*0
_output_shapes
:���������@@�*
T0�
/Cartoon_Generator/zero_padding2d_4/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             �
&Cartoon_Generator/zero_padding2d_4/PadPad1Cartoon_Generator/activation_4/Relu:activations:08Cartoon_Generator/zero_padding2d_4/Pad/paddings:output:0*0
_output_shapes
:���������BB�*
T0�
/Cartoon_Generator/conv5_2/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv5_2_conv2d_readvariableop_conv5_2_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv5_2/Conv2DConv2D/Cartoon_Generator/zero_padding2d_4/Pad:output:07Cartoon_Generator/conv5_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:���������@@��
0Cartoon_Generator/conv5_2/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv5_2_biasadd_readvariableop_conv5_2_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv5_2/BiasAddBiasAdd)Cartoon_Generator/conv5_2/Conv2D:output:08Cartoon_Generator/conv5_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm5_2/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm5_2/MeanMean*Cartoon_Generator/conv5_2/BiasAdd:output:09Cartoon_Generator/norm5_2/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
KCartoon_Generator/norm5_2/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm5_2/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv5_2/BiasAdd:output:0TCartoon_Generator/norm5_2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
8Cartoon_Generator/norm5_2/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv5_2/BiasAdd:output:0BCartoon_Generator/norm5_2/reduce_std/reduce_variance/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
;Cartoon_Generator/norm5_2/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm5_2/reduce_std/reduce_variance/sub:z:0*0
_output_shapes
:���������@@�*
T0�
MCartoon_Generator/norm5_2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         �
;Cartoon_Generator/norm5_2/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm5_2/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm5_2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
)Cartoon_Generator/norm5_2/reduce_std/SqrtSqrtDCartoon_Generator/norm5_2/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm5_2/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm5_2/addAdd-Cartoon_Generator/norm5_2/reduce_std/Sqrt:y:0(Cartoon_Generator/norm5_2/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm5_2/subSub*Cartoon_Generator/conv5_2/BiasAdd:output:0'Cartoon_Generator/norm5_2/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
!Cartoon_Generator/norm5_2/truedivRealDiv!Cartoon_Generator/norm5_2/sub:z:0!Cartoon_Generator/norm5_2/add:z:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm5_2/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm5_2_reshape_readvariableop_norm5_2_gamma*
_output_shapes
:*
dtype0�
'Cartoon_Generator/norm5_2/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm5_2/ReshapeReshape8Cartoon_Generator/norm5_2/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm5_2/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm5_2/mulMul%Cartoon_Generator/norm5_2/truediv:z:0*Cartoon_Generator/norm5_2/Reshape:output:0*0
_output_shapes
:���������@@�*
T0�
2Cartoon_Generator/norm5_2/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm5_2_reshape_1_readvariableop_norm5_2_beta*
_output_shapes
:*
dtype0�
)Cartoon_Generator/norm5_2/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm5_2/Reshape_1Reshape:Cartoon_Generator/norm5_2/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm5_2/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm5_2/add_1Add!Cartoon_Generator/norm5_2/mul:z:0,Cartoon_Generator/norm5_2/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
Cartoon_Generator/add_1/addAdd#Cartoon_Generator/norm5_2/add_1:z:0Cartoon_Generator/add/add:z:0*0
_output_shapes
:���������@@�*
T0�
/Cartoon_Generator/zero_padding2d_5/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             �
&Cartoon_Generator/zero_padding2d_5/PadPadCartoon_Generator/add_1/add:z:08Cartoon_Generator/zero_padding2d_5/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
/Cartoon_Generator/conv6_1/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv6_1_conv2d_readvariableop_conv6_1_kernel*(
_output_shapes
:��*
dtype0�
 Cartoon_Generator/conv6_1/Conv2DConv2D/Cartoon_Generator/zero_padding2d_5/Pad:output:07Cartoon_Generator/conv6_1/Conv2D/ReadVariableOp:value:0*
paddingVALID*0
_output_shapes
:���������@@�*
T0*
strides
�
0Cartoon_Generator/conv6_1/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv6_1_biasadd_readvariableop_conv6_1_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv6_1/BiasAddBiasAdd)Cartoon_Generator/conv6_1/Conv2D:output:08Cartoon_Generator/conv6_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm6_1/Mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0�
Cartoon_Generator/norm6_1/MeanMean*Cartoon_Generator/conv6_1/BiasAdd:output:09Cartoon_Generator/norm6_1/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
KCartoon_Generator/norm6_1/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0�
9Cartoon_Generator/norm6_1/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv6_1/BiasAdd:output:0TCartoon_Generator/norm6_1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
8Cartoon_Generator/norm6_1/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv6_1/BiasAdd:output:0BCartoon_Generator/norm6_1/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
;Cartoon_Generator/norm6_1/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm6_1/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
MCartoon_Generator/norm6_1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0�
;Cartoon_Generator/norm6_1/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm6_1/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm6_1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
)Cartoon_Generator/norm6_1/reduce_std/SqrtSqrtDCartoon_Generator/norm6_1/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm6_1/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm6_1/addAdd-Cartoon_Generator/norm6_1/reduce_std/Sqrt:y:0(Cartoon_Generator/norm6_1/add/y:output:0*/
_output_shapes
:���������*
T0�
Cartoon_Generator/norm6_1/subSub*Cartoon_Generator/conv6_1/BiasAdd:output:0'Cartoon_Generator/norm6_1/Mean:output:0*
T0*0
_output_shapes
:���������@@��
!Cartoon_Generator/norm6_1/truedivRealDiv!Cartoon_Generator/norm6_1/sub:z:0!Cartoon_Generator/norm6_1/add:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm6_1/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm6_1_reshape_readvariableop_norm6_1_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm6_1/Reshape/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0�
!Cartoon_Generator/norm6_1/ReshapeReshape8Cartoon_Generator/norm6_1/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm6_1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm6_1/mulMul%Cartoon_Generator/norm6_1/truediv:z:0*Cartoon_Generator/norm6_1/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm6_1/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm6_1_reshape_1_readvariableop_norm6_1_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm6_1/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm6_1/Reshape_1Reshape:Cartoon_Generator/norm6_1/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm6_1/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm6_1/add_1Add!Cartoon_Generator/norm6_1/mul:z:0,Cartoon_Generator/norm6_1/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
#Cartoon_Generator/activation_5/ReluRelu#Cartoon_Generator/norm6_1/add_1:z:0*
T0*0
_output_shapes
:���������@@��
/Cartoon_Generator/zero_padding2d_6/Pad/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0�
&Cartoon_Generator/zero_padding2d_6/PadPad1Cartoon_Generator/activation_5/Relu:activations:08Cartoon_Generator/zero_padding2d_6/Pad/paddings:output:0*0
_output_shapes
:���������BB�*
T0�
/Cartoon_Generator/conv6_2/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv6_2_conv2d_readvariableop_conv6_2_kernel*(
_output_shapes
:��*
dtype0�
 Cartoon_Generator/conv6_2/Conv2DConv2D/Cartoon_Generator/zero_padding2d_6/Pad:output:07Cartoon_Generator/conv6_2/Conv2D/ReadVariableOp:value:0*
paddingVALID*0
_output_shapes
:���������@@�*
T0*
strides
�
0Cartoon_Generator/conv6_2/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv6_2_biasadd_readvariableop_conv6_2_bias*
_output_shapes	
:�*
dtype0�
!Cartoon_Generator/conv6_2/BiasAddBiasAdd)Cartoon_Generator/conv6_2/Conv2D:output:08Cartoon_Generator/conv6_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm6_2/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm6_2/MeanMean*Cartoon_Generator/conv6_2/BiasAdd:output:09Cartoon_Generator/norm6_2/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
KCartoon_Generator/norm6_2/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm6_2/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv6_2/BiasAdd:output:0TCartoon_Generator/norm6_2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
8Cartoon_Generator/norm6_2/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv6_2/BiasAdd:output:0BCartoon_Generator/norm6_2/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
;Cartoon_Generator/norm6_2/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm6_2/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
MCartoon_Generator/norm6_2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm6_2/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm6_2/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm6_2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
)Cartoon_Generator/norm6_2/reduce_std/SqrtSqrtDCartoon_Generator/norm6_2/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm6_2/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm6_2/addAdd-Cartoon_Generator/norm6_2/reduce_std/Sqrt:y:0(Cartoon_Generator/norm6_2/add/y:output:0*/
_output_shapes
:���������*
T0�
Cartoon_Generator/norm6_2/subSub*Cartoon_Generator/conv6_2/BiasAdd:output:0'Cartoon_Generator/norm6_2/Mean:output:0*
T0*0
_output_shapes
:���������@@��
!Cartoon_Generator/norm6_2/truedivRealDiv!Cartoon_Generator/norm6_2/sub:z:0!Cartoon_Generator/norm6_2/add:z:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm6_2/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm6_2_reshape_readvariableop_norm6_2_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm6_2/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm6_2/ReshapeReshape8Cartoon_Generator/norm6_2/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm6_2/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm6_2/mulMul%Cartoon_Generator/norm6_2/truediv:z:0*Cartoon_Generator/norm6_2/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm6_2/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm6_2_reshape_1_readvariableop_norm6_2_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm6_2/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm6_2/Reshape_1Reshape:Cartoon_Generator/norm6_2/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm6_2/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm6_2/add_1Add!Cartoon_Generator/norm6_2/mul:z:0,Cartoon_Generator/norm6_2/Reshape_1:output:0*0
_output_shapes
:���������@@�*
T0�
Cartoon_Generator/add_2/addAdd#Cartoon_Generator/norm6_2/add_1:z:0Cartoon_Generator/add_1/add:z:0*
T0*0
_output_shapes
:���������@@��
/Cartoon_Generator/zero_padding2d_7/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             �
&Cartoon_Generator/zero_padding2d_7/PadPadCartoon_Generator/add_2/add:z:08Cartoon_Generator/zero_padding2d_7/Pad/paddings:output:0*0
_output_shapes
:���������BB�*
T0�
/Cartoon_Generator/conv7_1/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv7_1_conv2d_readvariableop_conv7_1_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv7_1/Conv2DConv2D/Cartoon_Generator/zero_padding2d_7/Pad:output:07Cartoon_Generator/conv7_1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/conv7_1/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv7_1_biasadd_readvariableop_conv7_1_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv7_1/BiasAddBiasAdd)Cartoon_Generator/conv7_1/Conv2D:output:08Cartoon_Generator/conv7_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm7_1/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm7_1/MeanMean*Cartoon_Generator/conv7_1/BiasAdd:output:09Cartoon_Generator/norm7_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
KCartoon_Generator/norm7_1/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm7_1/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv7_1/BiasAdd:output:0TCartoon_Generator/norm7_1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
8Cartoon_Generator/norm7_1/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv7_1/BiasAdd:output:0BCartoon_Generator/norm7_1/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
;Cartoon_Generator/norm7_1/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm7_1/reduce_std/reduce_variance/sub:z:0*0
_output_shapes
:���������@@�*
T0�
MCartoon_Generator/norm7_1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm7_1/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm7_1/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm7_1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
)Cartoon_Generator/norm7_1/reduce_std/SqrtSqrtDCartoon_Generator/norm7_1/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm7_1/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0�
Cartoon_Generator/norm7_1/addAdd-Cartoon_Generator/norm7_1/reduce_std/Sqrt:y:0(Cartoon_Generator/norm7_1/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm7_1/subSub*Cartoon_Generator/conv7_1/BiasAdd:output:0'Cartoon_Generator/norm7_1/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
!Cartoon_Generator/norm7_1/truedivRealDiv!Cartoon_Generator/norm7_1/sub:z:0!Cartoon_Generator/norm7_1/add:z:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm7_1/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm7_1_reshape_readvariableop_norm7_1_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm7_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm7_1/ReshapeReshape8Cartoon_Generator/norm7_1/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm7_1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm7_1/mulMul%Cartoon_Generator/norm7_1/truediv:z:0*Cartoon_Generator/norm7_1/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm7_1/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm7_1_reshape_1_readvariableop_norm7_1_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm7_1/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm7_1/Reshape_1Reshape:Cartoon_Generator/norm7_1/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm7_1/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm7_1/add_1Add!Cartoon_Generator/norm7_1/mul:z:0,Cartoon_Generator/norm7_1/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
#Cartoon_Generator/activation_6/ReluRelu#Cartoon_Generator/norm7_1/add_1:z:0*0
_output_shapes
:���������@@�*
T0�
/Cartoon_Generator/zero_padding2d_8/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
&Cartoon_Generator/zero_padding2d_8/PadPad1Cartoon_Generator/activation_6/Relu:activations:08Cartoon_Generator/zero_padding2d_8/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
/Cartoon_Generator/conv7_2/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv7_2_conv2d_readvariableop_conv7_2_kernel*(
_output_shapes
:��*
dtype0�
 Cartoon_Generator/conv7_2/Conv2DConv2D/Cartoon_Generator/zero_padding2d_8/Pad:output:07Cartoon_Generator/conv7_2/Conv2D/ReadVariableOp:value:0*
paddingVALID*0
_output_shapes
:���������@@�*
T0*
strides
�
0Cartoon_Generator/conv7_2/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv7_2_biasadd_readvariableop_conv7_2_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv7_2/BiasAddBiasAdd)Cartoon_Generator/conv7_2/Conv2D:output:08Cartoon_Generator/conv7_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm7_2/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm7_2/MeanMean*Cartoon_Generator/conv7_2/BiasAdd:output:09Cartoon_Generator/norm7_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
KCartoon_Generator/norm7_2/reduce_std/reduce_variance/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         �
9Cartoon_Generator/norm7_2/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv7_2/BiasAdd:output:0TCartoon_Generator/norm7_2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
8Cartoon_Generator/norm7_2/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv7_2/BiasAdd:output:0BCartoon_Generator/norm7_2/reduce_std/reduce_variance/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
;Cartoon_Generator/norm7_2/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm7_2/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
MCartoon_Generator/norm7_2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm7_2/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm7_2/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm7_2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
)Cartoon_Generator/norm7_2/reduce_std/SqrtSqrtDCartoon_Generator/norm7_2/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm7_2/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm7_2/addAdd-Cartoon_Generator/norm7_2/reduce_std/Sqrt:y:0(Cartoon_Generator/norm7_2/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm7_2/subSub*Cartoon_Generator/conv7_2/BiasAdd:output:0'Cartoon_Generator/norm7_2/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
!Cartoon_Generator/norm7_2/truedivRealDiv!Cartoon_Generator/norm7_2/sub:z:0!Cartoon_Generator/norm7_2/add:z:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm7_2/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm7_2_reshape_readvariableop_norm7_2_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm7_2/Reshape/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0�
!Cartoon_Generator/norm7_2/ReshapeReshape8Cartoon_Generator/norm7_2/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm7_2/Reshape/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm7_2/mulMul%Cartoon_Generator/norm7_2/truediv:z:0*Cartoon_Generator/norm7_2/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm7_2/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm7_2_reshape_1_readvariableop_norm7_2_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm7_2/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm7_2/Reshape_1Reshape:Cartoon_Generator/norm7_2/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm7_2/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm7_2/add_1Add!Cartoon_Generator/norm7_2/mul:z:0,Cartoon_Generator/norm7_2/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
Cartoon_Generator/add_3/addAdd#Cartoon_Generator/norm7_2/add_1:z:0Cartoon_Generator/add_2/add:z:0*
T0*0
_output_shapes
:���������@@��
/Cartoon_Generator/zero_padding2d_9/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
&Cartoon_Generator/zero_padding2d_9/PadPadCartoon_Generator/add_3/add:z:08Cartoon_Generator/zero_padding2d_9/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
/Cartoon_Generator/conv8_1/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv8_1_conv2d_readvariableop_conv8_1_kernel*(
_output_shapes
:��*
dtype0�
 Cartoon_Generator/conv8_1/Conv2DConv2D/Cartoon_Generator/zero_padding2d_9/Pad:output:07Cartoon_Generator/conv8_1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0*
strides
*
paddingVALID�
0Cartoon_Generator/conv8_1/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv8_1_biasadd_readvariableop_conv8_1_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv8_1/BiasAddBiasAdd)Cartoon_Generator/conv8_1/Conv2D:output:08Cartoon_Generator/conv8_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm8_1/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm8_1/MeanMean*Cartoon_Generator/conv8_1/BiasAdd:output:09Cartoon_Generator/norm8_1/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
KCartoon_Generator/norm8_1/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm8_1/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv8_1/BiasAdd:output:0TCartoon_Generator/norm8_1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
8Cartoon_Generator/norm8_1/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv8_1/BiasAdd:output:0BCartoon_Generator/norm8_1/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
;Cartoon_Generator/norm8_1/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm8_1/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
MCartoon_Generator/norm8_1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm8_1/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm8_1/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm8_1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
)Cartoon_Generator/norm8_1/reduce_std/SqrtSqrtDCartoon_Generator/norm8_1/reduce_std/reduce_variance/Mean_1:output:0*/
_output_shapes
:���������*
T0d
Cartoon_Generator/norm8_1/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm8_1/addAdd-Cartoon_Generator/norm8_1/reduce_std/Sqrt:y:0(Cartoon_Generator/norm8_1/add/y:output:0*/
_output_shapes
:���������*
T0�
Cartoon_Generator/norm8_1/subSub*Cartoon_Generator/conv8_1/BiasAdd:output:0'Cartoon_Generator/norm8_1/Mean:output:0*
T0*0
_output_shapes
:���������@@��
!Cartoon_Generator/norm8_1/truedivRealDiv!Cartoon_Generator/norm8_1/sub:z:0!Cartoon_Generator/norm8_1/add:z:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm8_1/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm8_1_reshape_readvariableop_norm8_1_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm8_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm8_1/ReshapeReshape8Cartoon_Generator/norm8_1/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm8_1/Reshape/shape:output:0*&
_output_shapes
:*
T0�
Cartoon_Generator/norm8_1/mulMul%Cartoon_Generator/norm8_1/truediv:z:0*Cartoon_Generator/norm8_1/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm8_1/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm8_1_reshape_1_readvariableop_norm8_1_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm8_1/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm8_1/Reshape_1Reshape:Cartoon_Generator/norm8_1/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm8_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm8_1/add_1Add!Cartoon_Generator/norm8_1/mul:z:0,Cartoon_Generator/norm8_1/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
#Cartoon_Generator/activation_7/ReluRelu#Cartoon_Generator/norm8_1/add_1:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/zero_padding2d_10/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
'Cartoon_Generator/zero_padding2d_10/PadPad1Cartoon_Generator/activation_7/Relu:activations:09Cartoon_Generator/zero_padding2d_10/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
/Cartoon_Generator/conv8_2/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv8_2_conv2d_readvariableop_conv8_2_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv8_2/Conv2DConv2D0Cartoon_Generator/zero_padding2d_10/Pad:output:07Cartoon_Generator/conv8_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:���������@@��
0Cartoon_Generator/conv8_2/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv8_2_biasadd_readvariableop_conv8_2_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv8_2/BiasAddBiasAdd)Cartoon_Generator/conv8_2/Conv2D:output:08Cartoon_Generator/conv8_2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm8_2/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm8_2/MeanMean*Cartoon_Generator/conv8_2/BiasAdd:output:09Cartoon_Generator/norm8_2/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
KCartoon_Generator/norm8_2/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm8_2/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv8_2/BiasAdd:output:0TCartoon_Generator/norm8_2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
8Cartoon_Generator/norm8_2/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv8_2/BiasAdd:output:0BCartoon_Generator/norm8_2/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
;Cartoon_Generator/norm8_2/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm8_2/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
MCartoon_Generator/norm8_2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm8_2/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm8_2/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm8_2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
)Cartoon_Generator/norm8_2/reduce_std/SqrtSqrtDCartoon_Generator/norm8_2/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm8_2/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm8_2/addAdd-Cartoon_Generator/norm8_2/reduce_std/Sqrt:y:0(Cartoon_Generator/norm8_2/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm8_2/subSub*Cartoon_Generator/conv8_2/BiasAdd:output:0'Cartoon_Generator/norm8_2/Mean:output:0*
T0*0
_output_shapes
:���������@@��
!Cartoon_Generator/norm8_2/truedivRealDiv!Cartoon_Generator/norm8_2/sub:z:0!Cartoon_Generator/norm8_2/add:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm8_2/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm8_2_reshape_readvariableop_norm8_2_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm8_2/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm8_2/ReshapeReshape8Cartoon_Generator/norm8_2/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm8_2/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm8_2/mulMul%Cartoon_Generator/norm8_2/truediv:z:0*Cartoon_Generator/norm8_2/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm8_2/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm8_2_reshape_1_readvariableop_norm8_2_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm8_2/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm8_2/Reshape_1Reshape:Cartoon_Generator/norm8_2/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm8_2/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm8_2/add_1Add!Cartoon_Generator/norm8_2/mul:z:0,Cartoon_Generator/norm8_2/Reshape_1:output:0*0
_output_shapes
:���������@@�*
T0�
Cartoon_Generator/add_4/addAdd#Cartoon_Generator/norm8_2/add_1:z:0Cartoon_Generator/add_3/add:z:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/zero_padding2d_11/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
'Cartoon_Generator/zero_padding2d_11/PadPadCartoon_Generator/add_4/add:z:09Cartoon_Generator/zero_padding2d_11/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
/Cartoon_Generator/conv9_1/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv9_1_conv2d_readvariableop_conv9_1_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv9_1/Conv2DConv2D0Cartoon_Generator/zero_padding2d_11/Pad:output:07Cartoon_Generator/conv9_1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/conv9_1/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv9_1_biasadd_readvariableop_conv9_1_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv9_1/BiasAddBiasAdd)Cartoon_Generator/conv9_1/Conv2D:output:08Cartoon_Generator/conv9_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm9_1/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm9_1/MeanMean*Cartoon_Generator/conv9_1/BiasAdd:output:09Cartoon_Generator/norm9_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
KCartoon_Generator/norm9_1/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm9_1/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv9_1/BiasAdd:output:0TCartoon_Generator/norm9_1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
8Cartoon_Generator/norm9_1/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv9_1/BiasAdd:output:0BCartoon_Generator/norm9_1/reduce_std/reduce_variance/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
;Cartoon_Generator/norm9_1/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm9_1/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
MCartoon_Generator/norm9_1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm9_1/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm9_1/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm9_1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
)Cartoon_Generator/norm9_1/reduce_std/SqrtSqrtDCartoon_Generator/norm9_1/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm9_1/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm9_1/addAdd-Cartoon_Generator/norm9_1/reduce_std/Sqrt:y:0(Cartoon_Generator/norm9_1/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm9_1/subSub*Cartoon_Generator/conv9_1/BiasAdd:output:0'Cartoon_Generator/norm9_1/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
!Cartoon_Generator/norm9_1/truedivRealDiv!Cartoon_Generator/norm9_1/sub:z:0!Cartoon_Generator/norm9_1/add:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm9_1/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm9_1_reshape_readvariableop_norm9_1_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm9_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
!Cartoon_Generator/norm9_1/ReshapeReshape8Cartoon_Generator/norm9_1/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm9_1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm9_1/mulMul%Cartoon_Generator/norm9_1/truediv:z:0*Cartoon_Generator/norm9_1/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
2Cartoon_Generator/norm9_1/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm9_1_reshape_1_readvariableop_norm9_1_beta*
_output_shapes
:*
dtype0�
)Cartoon_Generator/norm9_1/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm9_1/Reshape_1Reshape:Cartoon_Generator/norm9_1/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm9_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm9_1/add_1Add!Cartoon_Generator/norm9_1/mul:z:0,Cartoon_Generator/norm9_1/Reshape_1:output:0*0
_output_shapes
:���������@@�*
T0�
#Cartoon_Generator/activation_8/ReluRelu#Cartoon_Generator/norm9_1/add_1:z:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/zero_padding2d_12/Pad/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0�
'Cartoon_Generator/zero_padding2d_12/PadPad1Cartoon_Generator/activation_8/Relu:activations:09Cartoon_Generator/zero_padding2d_12/Pad/paddings:output:0*0
_output_shapes
:���������BB�*
T0�
/Cartoon_Generator/conv9_2/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_conv9_2_conv2d_readvariableop_conv9_2_kernel*
dtype0*(
_output_shapes
:���
 Cartoon_Generator/conv9_2/Conv2DConv2D0Cartoon_Generator/zero_padding2d_12/Pad:output:07Cartoon_Generator/conv9_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:���������@@��
0Cartoon_Generator/conv9_2/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_conv9_2_biasadd_readvariableop_conv9_2_bias*
dtype0*
_output_shapes	
:��
!Cartoon_Generator/conv9_2/BiasAddBiasAdd)Cartoon_Generator/conv9_2/Conv2D:output:08Cartoon_Generator/conv9_2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0�
0Cartoon_Generator/norm9_2/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm9_2/MeanMean*Cartoon_Generator/conv9_2/BiasAdd:output:09Cartoon_Generator/norm9_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
KCartoon_Generator/norm9_2/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
9Cartoon_Generator/norm9_2/reduce_std/reduce_variance/MeanMean*Cartoon_Generator/conv9_2/BiasAdd:output:0TCartoon_Generator/norm9_2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
8Cartoon_Generator/norm9_2/reduce_std/reduce_variance/subSub*Cartoon_Generator/conv9_2/BiasAdd:output:0BCartoon_Generator/norm9_2/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
;Cartoon_Generator/norm9_2/reduce_std/reduce_variance/SquareSquare<Cartoon_Generator/norm9_2/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
MCartoon_Generator/norm9_2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
;Cartoon_Generator/norm9_2/reduce_std/reduce_variance/Mean_1Mean?Cartoon_Generator/norm9_2/reduce_std/reduce_variance/Square:y:0VCartoon_Generator/norm9_2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
)Cartoon_Generator/norm9_2/reduce_std/SqrtSqrtDCartoon_Generator/norm9_2/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������d
Cartoon_Generator/norm9_2/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm9_2/addAdd-Cartoon_Generator/norm9_2/reduce_std/Sqrt:y:0(Cartoon_Generator/norm9_2/add/y:output:0*/
_output_shapes
:���������*
T0�
Cartoon_Generator/norm9_2/subSub*Cartoon_Generator/conv9_2/BiasAdd:output:0'Cartoon_Generator/norm9_2/Mean:output:0*
T0*0
_output_shapes
:���������@@��
!Cartoon_Generator/norm9_2/truedivRealDiv!Cartoon_Generator/norm9_2/sub:z:0!Cartoon_Generator/norm9_2/add:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/norm9_2/Reshape/ReadVariableOpReadVariableOp>cartoon_generator_norm9_2_reshape_readvariableop_norm9_2_gamma*
dtype0*
_output_shapes
:�
'Cartoon_Generator/norm9_2/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            �
!Cartoon_Generator/norm9_2/ReshapeReshape8Cartoon_Generator/norm9_2/Reshape/ReadVariableOp:value:00Cartoon_Generator/norm9_2/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm9_2/mulMul%Cartoon_Generator/norm9_2/truediv:z:0*Cartoon_Generator/norm9_2/Reshape:output:0*0
_output_shapes
:���������@@�*
T0�
2Cartoon_Generator/norm9_2/Reshape_1/ReadVariableOpReadVariableOp?cartoon_generator_norm9_2_reshape_1_readvariableop_norm9_2_beta*
dtype0*
_output_shapes
:�
)Cartoon_Generator/norm9_2/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            �
#Cartoon_Generator/norm9_2/Reshape_1Reshape:Cartoon_Generator/norm9_2/Reshape_1/ReadVariableOp:value:02Cartoon_Generator/norm9_2/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm9_2/add_1Add!Cartoon_Generator/norm9_2/mul:z:0,Cartoon_Generator/norm9_2/Reshape_1:output:0*0
_output_shapes
:���������@@�*
T0�
Cartoon_Generator/add_5/addAdd#Cartoon_Generator/norm9_2/add_1:z:0Cartoon_Generator/add_4/add:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/zero_padding2d_13/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
'Cartoon_Generator/zero_padding2d_13/PadPadCartoon_Generator/add_5/add:z:09Cartoon_Generator/zero_padding2d_13/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
0Cartoon_Generator/conv10_1/Conv2D/ReadVariableOpReadVariableOp@cartoon_generator_conv10_1_conv2d_readvariableop_conv10_1_kernel*(
_output_shapes
:��*
dtype0�
!Cartoon_Generator/conv10_1/Conv2DConv2D0Cartoon_Generator/zero_padding2d_13/Pad:output:08Cartoon_Generator/conv10_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:���������@@��
1Cartoon_Generator/conv10_1/BiasAdd/ReadVariableOpReadVariableOp?cartoon_generator_conv10_1_biasadd_readvariableop_conv10_1_bias*
dtype0*
_output_shapes	
:��
"Cartoon_Generator/conv10_1/BiasAddBiasAdd*Cartoon_Generator/conv10_1/Conv2D:output:09Cartoon_Generator/conv10_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:���������@@�*
T0�
1Cartoon_Generator/norm10_1/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm10_1/MeanMean+Cartoon_Generator/conv10_1/BiasAdd:output:0:Cartoon_Generator/norm10_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
LCartoon_Generator/norm10_1/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
:Cartoon_Generator/norm10_1/reduce_std/reduce_variance/MeanMean+Cartoon_Generator/conv10_1/BiasAdd:output:0UCartoon_Generator/norm10_1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
9Cartoon_Generator/norm10_1/reduce_std/reduce_variance/subSub+Cartoon_Generator/conv10_1/BiasAdd:output:0CCartoon_Generator/norm10_1/reduce_std/reduce_variance/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
<Cartoon_Generator/norm10_1/reduce_std/reduce_variance/SquareSquare=Cartoon_Generator/norm10_1/reduce_std/reduce_variance/sub:z:0*0
_output_shapes
:���������@@�*
T0�
NCartoon_Generator/norm10_1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0�
<Cartoon_Generator/norm10_1/reduce_std/reduce_variance/Mean_1Mean@Cartoon_Generator/norm10_1/reduce_std/reduce_variance/Square:y:0WCartoon_Generator/norm10_1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
*Cartoon_Generator/norm10_1/reduce_std/SqrtSqrtECartoon_Generator/norm10_1/reduce_std/reduce_variance/Mean_1:output:0*/
_output_shapes
:���������*
T0e
 Cartoon_Generator/norm10_1/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0�
Cartoon_Generator/norm10_1/addAdd.Cartoon_Generator/norm10_1/reduce_std/Sqrt:y:0)Cartoon_Generator/norm10_1/add/y:output:0*/
_output_shapes
:���������*
T0�
Cartoon_Generator/norm10_1/subSub+Cartoon_Generator/conv10_1/BiasAdd:output:0(Cartoon_Generator/norm10_1/Mean:output:0*
T0*0
_output_shapes
:���������@@��
"Cartoon_Generator/norm10_1/truedivRealDiv"Cartoon_Generator/norm10_1/sub:z:0"Cartoon_Generator/norm10_1/add:z:0*
T0*0
_output_shapes
:���������@@��
1Cartoon_Generator/norm10_1/Reshape/ReadVariableOpReadVariableOp@cartoon_generator_norm10_1_reshape_readvariableop_norm10_1_gamma*
dtype0*
_output_shapes
:�
(Cartoon_Generator/norm10_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
"Cartoon_Generator/norm10_1/ReshapeReshape9Cartoon_Generator/norm10_1/Reshape/ReadVariableOp:value:01Cartoon_Generator/norm10_1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm10_1/mulMul&Cartoon_Generator/norm10_1/truediv:z:0+Cartoon_Generator/norm10_1/Reshape:output:0*0
_output_shapes
:���������@@�*
T0�
3Cartoon_Generator/norm10_1/Reshape_1/ReadVariableOpReadVariableOpAcartoon_generator_norm10_1_reshape_1_readvariableop_norm10_1_beta*
dtype0*
_output_shapes
:�
*Cartoon_Generator/norm10_1/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
$Cartoon_Generator/norm10_1/Reshape_1Reshape;Cartoon_Generator/norm10_1/Reshape_1/ReadVariableOp:value:03Cartoon_Generator/norm10_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
 Cartoon_Generator/norm10_1/add_1Add"Cartoon_Generator/norm10_1/mul:z:0-Cartoon_Generator/norm10_1/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
#Cartoon_Generator/activation_9/ReluRelu$Cartoon_Generator/norm10_1/add_1:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/zero_padding2d_14/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
'Cartoon_Generator/zero_padding2d_14/PadPad1Cartoon_Generator/activation_9/Relu:activations:09Cartoon_Generator/zero_padding2d_14/Pad/paddings:output:0*0
_output_shapes
:���������BB�*
T0�
0Cartoon_Generator/conv10_2/Conv2D/ReadVariableOpReadVariableOp@cartoon_generator_conv10_2_conv2d_readvariableop_conv10_2_kernel*(
_output_shapes
:��*
dtype0�
!Cartoon_Generator/conv10_2/Conv2DConv2D0Cartoon_Generator/zero_padding2d_14/Pad:output:08Cartoon_Generator/conv10_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:���������@@��
1Cartoon_Generator/conv10_2/BiasAdd/ReadVariableOpReadVariableOp?cartoon_generator_conv10_2_biasadd_readvariableop_conv10_2_bias*
dtype0*
_output_shapes	
:��
"Cartoon_Generator/conv10_2/BiasAddBiasAdd*Cartoon_Generator/conv10_2/Conv2D:output:09Cartoon_Generator/conv10_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
1Cartoon_Generator/norm10_2/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         �
Cartoon_Generator/norm10_2/MeanMean+Cartoon_Generator/conv10_2/BiasAdd:output:0:Cartoon_Generator/norm10_2/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
LCartoon_Generator/norm10_2/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
:Cartoon_Generator/norm10_2/reduce_std/reduce_variance/MeanMean+Cartoon_Generator/conv10_2/BiasAdd:output:0UCartoon_Generator/norm10_2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
9Cartoon_Generator/norm10_2/reduce_std/reduce_variance/subSub+Cartoon_Generator/conv10_2/BiasAdd:output:0CCartoon_Generator/norm10_2/reduce_std/reduce_variance/Mean:output:0*0
_output_shapes
:���������@@�*
T0�
<Cartoon_Generator/norm10_2/reduce_std/reduce_variance/SquareSquare=Cartoon_Generator/norm10_2/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
NCartoon_Generator/norm10_2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
<Cartoon_Generator/norm10_2/reduce_std/reduce_variance/Mean_1Mean@Cartoon_Generator/norm10_2/reduce_std/reduce_variance/Square:y:0WCartoon_Generator/norm10_2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
*Cartoon_Generator/norm10_2/reduce_std/SqrtSqrtECartoon_Generator/norm10_2/reduce_std/reduce_variance/Mean_1:output:0*/
_output_shapes
:���������*
T0e
 Cartoon_Generator/norm10_2/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:�
Cartoon_Generator/norm10_2/addAdd.Cartoon_Generator/norm10_2/reduce_std/Sqrt:y:0)Cartoon_Generator/norm10_2/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm10_2/subSub+Cartoon_Generator/conv10_2/BiasAdd:output:0(Cartoon_Generator/norm10_2/Mean:output:0*
T0*0
_output_shapes
:���������@@��
"Cartoon_Generator/norm10_2/truedivRealDiv"Cartoon_Generator/norm10_2/sub:z:0"Cartoon_Generator/norm10_2/add:z:0*0
_output_shapes
:���������@@�*
T0�
1Cartoon_Generator/norm10_2/Reshape/ReadVariableOpReadVariableOp@cartoon_generator_norm10_2_reshape_readvariableop_norm10_2_gamma*
dtype0*
_output_shapes
:�
(Cartoon_Generator/norm10_2/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
"Cartoon_Generator/norm10_2/ReshapeReshape9Cartoon_Generator/norm10_2/Reshape/ReadVariableOp:value:01Cartoon_Generator/norm10_2/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm10_2/mulMul&Cartoon_Generator/norm10_2/truediv:z:0+Cartoon_Generator/norm10_2/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
3Cartoon_Generator/norm10_2/Reshape_1/ReadVariableOpReadVariableOpAcartoon_generator_norm10_2_reshape_1_readvariableop_norm10_2_beta*
dtype0*
_output_shapes
:�
*Cartoon_Generator/norm10_2/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
$Cartoon_Generator/norm10_2/Reshape_1Reshape;Cartoon_Generator/norm10_2/Reshape_1/ReadVariableOp:value:03Cartoon_Generator/norm10_2/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
 Cartoon_Generator/norm10_2/add_1Add"Cartoon_Generator/norm10_2/mul:z:0-Cartoon_Generator/norm10_2/Reshape_1:output:0*
T0*0
_output_shapes
:���������@@��
Cartoon_Generator/add_6/addAdd$Cartoon_Generator/norm10_2/add_1:z:0Cartoon_Generator/add_5/add:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/zero_padding2d_15/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
'Cartoon_Generator/zero_padding2d_15/PadPadCartoon_Generator/add_6/add:z:09Cartoon_Generator/zero_padding2d_15/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
0Cartoon_Generator/conv11_1/Conv2D/ReadVariableOpReadVariableOp@cartoon_generator_conv11_1_conv2d_readvariableop_conv11_1_kernel*
dtype0*(
_output_shapes
:���
!Cartoon_Generator/conv11_1/Conv2DConv2D0Cartoon_Generator/zero_padding2d_15/Pad:output:08Cartoon_Generator/conv11_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:���������@@��
1Cartoon_Generator/conv11_1/BiasAdd/ReadVariableOpReadVariableOp?cartoon_generator_conv11_1_biasadd_readvariableop_conv11_1_bias*
dtype0*
_output_shapes	
:��
"Cartoon_Generator/conv11_1/BiasAddBiasAdd*Cartoon_Generator/conv11_1/Conv2D:output:09Cartoon_Generator/conv11_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
1Cartoon_Generator/norm11_1/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm11_1/MeanMean+Cartoon_Generator/conv11_1/BiasAdd:output:0:Cartoon_Generator/norm11_1/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
LCartoon_Generator/norm11_1/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
:Cartoon_Generator/norm11_1/reduce_std/reduce_variance/MeanMean+Cartoon_Generator/conv11_1/BiasAdd:output:0UCartoon_Generator/norm11_1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*/
_output_shapes
:���������*
	keep_dims(*
T0�
9Cartoon_Generator/norm11_1/reduce_std/reduce_variance/subSub+Cartoon_Generator/conv11_1/BiasAdd:output:0CCartoon_Generator/norm11_1/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
<Cartoon_Generator/norm11_1/reduce_std/reduce_variance/SquareSquare=Cartoon_Generator/norm11_1/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:���������@@��
NCartoon_Generator/norm11_1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
<Cartoon_Generator/norm11_1/reduce_std/reduce_variance/Mean_1Mean@Cartoon_Generator/norm11_1/reduce_std/reduce_variance/Square:y:0WCartoon_Generator/norm11_1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
*Cartoon_Generator/norm11_1/reduce_std/SqrtSqrtECartoon_Generator/norm11_1/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������e
 Cartoon_Generator/norm11_1/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm11_1/addAdd.Cartoon_Generator/norm11_1/reduce_std/Sqrt:y:0)Cartoon_Generator/norm11_1/add/y:output:0*
T0*/
_output_shapes
:����������
Cartoon_Generator/norm11_1/subSub+Cartoon_Generator/conv11_1/BiasAdd:output:0(Cartoon_Generator/norm11_1/Mean:output:0*
T0*0
_output_shapes
:���������@@��
"Cartoon_Generator/norm11_1/truedivRealDiv"Cartoon_Generator/norm11_1/sub:z:0"Cartoon_Generator/norm11_1/add:z:0*
T0*0
_output_shapes
:���������@@��
1Cartoon_Generator/norm11_1/Reshape/ReadVariableOpReadVariableOp@cartoon_generator_norm11_1_reshape_readvariableop_norm11_1_gamma*
dtype0*
_output_shapes
:�
(Cartoon_Generator/norm11_1/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
"Cartoon_Generator/norm11_1/ReshapeReshape9Cartoon_Generator/norm11_1/Reshape/ReadVariableOp:value:01Cartoon_Generator/norm11_1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm11_1/mulMul&Cartoon_Generator/norm11_1/truediv:z:0+Cartoon_Generator/norm11_1/Reshape:output:0*
T0*0
_output_shapes
:���������@@��
3Cartoon_Generator/norm11_1/Reshape_1/ReadVariableOpReadVariableOpAcartoon_generator_norm11_1_reshape_1_readvariableop_norm11_1_beta*
dtype0*
_output_shapes
:�
*Cartoon_Generator/norm11_1/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
$Cartoon_Generator/norm11_1/Reshape_1Reshape;Cartoon_Generator/norm11_1/Reshape_1/ReadVariableOp:value:03Cartoon_Generator/norm11_1/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
 Cartoon_Generator/norm11_1/add_1Add"Cartoon_Generator/norm11_1/mul:z:0-Cartoon_Generator/norm11_1/Reshape_1:output:0*0
_output_shapes
:���������@@�*
T0�
$Cartoon_Generator/activation_10/ReluRelu$Cartoon_Generator/norm11_1/add_1:z:0*
T0*0
_output_shapes
:���������@@��
0Cartoon_Generator/zero_padding2d_16/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             �
'Cartoon_Generator/zero_padding2d_16/PadPad2Cartoon_Generator/activation_10/Relu:activations:09Cartoon_Generator/zero_padding2d_16/Pad/paddings:output:0*
T0*0
_output_shapes
:���������BB��
0Cartoon_Generator/conv11_2/Conv2D/ReadVariableOpReadVariableOp@cartoon_generator_conv11_2_conv2d_readvariableop_conv11_2_kernel*
dtype0*(
_output_shapes
:���
!Cartoon_Generator/conv11_2/Conv2DConv2D0Cartoon_Generator/zero_padding2d_16/Pad:output:08Cartoon_Generator/conv11_2/Conv2D/ReadVariableOp:value:0*
paddingVALID*0
_output_shapes
:���������@@�*
T0*
strides
�
1Cartoon_Generator/conv11_2/BiasAdd/ReadVariableOpReadVariableOp?cartoon_generator_conv11_2_biasadd_readvariableop_conv11_2_bias*
dtype0*
_output_shapes	
:��
"Cartoon_Generator/conv11_2/BiasAddBiasAdd*Cartoon_Generator/conv11_2/Conv2D:output:09Cartoon_Generator/conv11_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������@@��
1Cartoon_Generator/norm11_2/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
Cartoon_Generator/norm11_2/MeanMean+Cartoon_Generator/conv11_2/BiasAdd:output:0:Cartoon_Generator/norm11_2/Mean/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
LCartoon_Generator/norm11_2/reduce_std/reduce_variance/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         �
:Cartoon_Generator/norm11_2/reduce_std/reduce_variance/MeanMean+Cartoon_Generator/conv11_2/BiasAdd:output:0UCartoon_Generator/norm11_2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
9Cartoon_Generator/norm11_2/reduce_std/reduce_variance/subSub+Cartoon_Generator/conv11_2/BiasAdd:output:0CCartoon_Generator/norm11_2/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:���������@@��
<Cartoon_Generator/norm11_2/reduce_std/reduce_variance/SquareSquare=Cartoon_Generator/norm11_2/reduce_std/reduce_variance/sub:z:0*0
_output_shapes
:���������@@�*
T0�
NCartoon_Generator/norm11_2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         �
<Cartoon_Generator/norm11_2/reduce_std/reduce_variance/Mean_1Mean@Cartoon_Generator/norm11_2/reduce_std/reduce_variance/Square:y:0WCartoon_Generator/norm11_2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
*Cartoon_Generator/norm11_2/reduce_std/SqrtSqrtECartoon_Generator/norm11_2/reduce_std/reduce_variance/Mean_1:output:0*/
_output_shapes
:���������*
T0e
 Cartoon_Generator/norm11_2/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
Cartoon_Generator/norm11_2/addAdd.Cartoon_Generator/norm11_2/reduce_std/Sqrt:y:0)Cartoon_Generator/norm11_2/add/y:output:0*/
_output_shapes
:���������*
T0�
Cartoon_Generator/norm11_2/subSub+Cartoon_Generator/conv11_2/BiasAdd:output:0(Cartoon_Generator/norm11_2/Mean:output:0*
T0*0
_output_shapes
:���������@@��
"Cartoon_Generator/norm11_2/truedivRealDiv"Cartoon_Generator/norm11_2/sub:z:0"Cartoon_Generator/norm11_2/add:z:0*0
_output_shapes
:���������@@�*
T0�
1Cartoon_Generator/norm11_2/Reshape/ReadVariableOpReadVariableOp@cartoon_generator_norm11_2_reshape_readvariableop_norm11_2_gamma*
dtype0*
_output_shapes
:�
(Cartoon_Generator/norm11_2/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
"Cartoon_Generator/norm11_2/ReshapeReshape9Cartoon_Generator/norm11_2/Reshape/ReadVariableOp:value:01Cartoon_Generator/norm11_2/Reshape/shape:output:0*
T0*&
_output_shapes
:�
Cartoon_Generator/norm11_2/mulMul&Cartoon_Generator/norm11_2/truediv:z:0+Cartoon_Generator/norm11_2/Reshape:output:0*0
_output_shapes
:���������@@�*
T0�
3Cartoon_Generator/norm11_2/Reshape_1/ReadVariableOpReadVariableOpAcartoon_generator_norm11_2_reshape_1_readvariableop_norm11_2_beta*
dtype0*
_output_shapes
:�
*Cartoon_Generator/norm11_2/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
$Cartoon_Generator/norm11_2/Reshape_1Reshape;Cartoon_Generator/norm11_2/Reshape_1/ReadVariableOp:value:03Cartoon_Generator/norm11_2/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
 Cartoon_Generator/norm11_2/add_1Add"Cartoon_Generator/norm11_2/mul:z:0-Cartoon_Generator/norm11_2/Reshape_1:output:0*0
_output_shapes
:���������@@�*
T0�
Cartoon_Generator/add_7/addAdd$Cartoon_Generator/norm11_2/add_1:z:0Cartoon_Generator/add_6/add:z:0*0
_output_shapes
:���������@@�*
T0p
!Cartoon_Generator/deconv1_1/ShapeShapeCartoon_Generator/add_7/add:z:0*
T0*
_output_shapes
:y
/Cartoon_Generator/deconv1_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:{
1Cartoon_Generator/deconv1_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:{
1Cartoon_Generator/deconv1_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
)Cartoon_Generator/deconv1_1/strided_sliceStridedSlice*Cartoon_Generator/deconv1_1/Shape:output:08Cartoon_Generator/deconv1_1/strided_slice/stack:output:0:Cartoon_Generator/deconv1_1/strided_slice/stack_1:output:0:Cartoon_Generator/deconv1_1/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: {
1Cartoon_Generator/deconv1_1/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0}
3Cartoon_Generator/deconv1_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:}
3Cartoon_Generator/deconv1_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
+Cartoon_Generator/deconv1_1/strided_slice_1StridedSlice*Cartoon_Generator/deconv1_1/Shape:output:0:Cartoon_Generator/deconv1_1/strided_slice_1/stack:output:0<Cartoon_Generator/deconv1_1/strided_slice_1/stack_1:output:0<Cartoon_Generator/deconv1_1/strided_slice_1/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask{
1Cartoon_Generator/deconv1_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:}
3Cartoon_Generator/deconv1_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:}
3Cartoon_Generator/deconv1_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
+Cartoon_Generator/deconv1_1/strided_slice_2StridedSlice*Cartoon_Generator/deconv1_1/Shape:output:0:Cartoon_Generator/deconv1_1/strided_slice_2/stack:output:0<Cartoon_Generator/deconv1_1/strided_slice_2/stack_1:output:0<Cartoon_Generator/deconv1_1/strided_slice_2/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0c
!Cartoon_Generator/deconv1_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: �
Cartoon_Generator/deconv1_1/mulMul4Cartoon_Generator/deconv1_1/strided_slice_1:output:0*Cartoon_Generator/deconv1_1/mul/y:output:0*
T0*
_output_shapes
: e
#Cartoon_Generator/deconv1_1/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: �
!Cartoon_Generator/deconv1_1/mul_1Mul4Cartoon_Generator/deconv1_1/strided_slice_2:output:0,Cartoon_Generator/deconv1_1/mul_1/y:output:0*
T0*
_output_shapes
: f
#Cartoon_Generator/deconv1_1/stack/3Const*
value
B :�*
dtype0*
_output_shapes
: �
!Cartoon_Generator/deconv1_1/stackPack2Cartoon_Generator/deconv1_1/strided_slice:output:0#Cartoon_Generator/deconv1_1/mul:z:0%Cartoon_Generator/deconv1_1/mul_1:z:0,Cartoon_Generator/deconv1_1/stack/3:output:0*
_output_shapes
:*
T0*
N{
1Cartoon_Generator/deconv1_1/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:}
3Cartoon_Generator/deconv1_1/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:}
3Cartoon_Generator/deconv1_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
+Cartoon_Generator/deconv1_1/strided_slice_3StridedSlice*Cartoon_Generator/deconv1_1/stack:output:0:Cartoon_Generator/deconv1_1/strided_slice_3/stack:output:0<Cartoon_Generator/deconv1_1/strided_slice_3/stack_1:output:0<Cartoon_Generator/deconv1_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: �
;Cartoon_Generator/deconv1_1/conv2d_transpose/ReadVariableOpReadVariableOpLcartoon_generator_deconv1_1_conv2d_transpose_readvariableop_deconv1_1_kernel*
dtype0*(
_output_shapes
:���
,Cartoon_Generator/deconv1_1/conv2d_transposeConv2DBackpropInput*Cartoon_Generator/deconv1_1/stack:output:0CCartoon_Generator/deconv1_1/conv2d_transpose/ReadVariableOp:value:0Cartoon_Generator/add_7/add:z:0*2
_output_shapes 
:������������*
T0*
strides
*
paddingSAME�
2Cartoon_Generator/deconv1_1/BiasAdd/ReadVariableOpReadVariableOpAcartoon_generator_deconv1_1_biasadd_readvariableop_deconv1_1_bias*
dtype0*
_output_shapes	
:��
#Cartoon_Generator/deconv1_1/BiasAddBiasAdd5Cartoon_Generator/deconv1_1/conv2d_transpose:output:0:Cartoon_Generator/deconv1_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:�������������
1Cartoon_Generator/deconv1_2/Conv2D/ReadVariableOpReadVariableOpBcartoon_generator_deconv1_2_conv2d_readvariableop_deconv1_2_kernel*
dtype0*(
_output_shapes
:���
"Cartoon_Generator/deconv1_2/Conv2DConv2D,Cartoon_Generator/deconv1_1/BiasAdd:output:09Cartoon_Generator/deconv1_2/Conv2D/ReadVariableOp:value:0*2
_output_shapes 
:������������*
T0*
strides
*
paddingSAME�
2Cartoon_Generator/deconv1_2/BiasAdd/ReadVariableOpReadVariableOpAcartoon_generator_deconv1_2_biasadd_readvariableop_deconv1_2_bias*
dtype0*
_output_shapes	
:��
#Cartoon_Generator/deconv1_2/BiasAddBiasAdd+Cartoon_Generator/deconv1_2/Conv2D:output:0:Cartoon_Generator/deconv1_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:�������������
5Cartoon_Generator/norm_deconv1/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
#Cartoon_Generator/norm_deconv1/MeanMean,Cartoon_Generator/deconv1_2/BiasAdd:output:0>Cartoon_Generator/norm_deconv1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
PCartoon_Generator/norm_deconv1/reduce_std/reduce_variance/Mean/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
>Cartoon_Generator/norm_deconv1/reduce_std/reduce_variance/MeanMean,Cartoon_Generator/deconv1_2/BiasAdd:output:0YCartoon_Generator/norm_deconv1/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
=Cartoon_Generator/norm_deconv1/reduce_std/reduce_variance/subSub,Cartoon_Generator/deconv1_2/BiasAdd:output:0GCartoon_Generator/norm_deconv1/reduce_std/reduce_variance/Mean:output:0*
T0*2
_output_shapes 
:�������������
@Cartoon_Generator/norm_deconv1/reduce_std/reduce_variance/SquareSquareACartoon_Generator/norm_deconv1/reduce_std/reduce_variance/sub:z:0*
T0*2
_output_shapes 
:�������������
RCartoon_Generator/norm_deconv1/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"         �
@Cartoon_Generator/norm_deconv1/reduce_std/reduce_variance/Mean_1MeanDCartoon_Generator/norm_deconv1/reduce_std/reduce_variance/Square:y:0[Cartoon_Generator/norm_deconv1/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
	keep_dims(*
T0*/
_output_shapes
:����������
.Cartoon_Generator/norm_deconv1/reduce_std/SqrtSqrtICartoon_Generator/norm_deconv1/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������i
$Cartoon_Generator/norm_deconv1/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0�
"Cartoon_Generator/norm_deconv1/addAdd2Cartoon_Generator/norm_deconv1/reduce_std/Sqrt:y:0-Cartoon_Generator/norm_deconv1/add/y:output:0*
T0*/
_output_shapes
:����������
"Cartoon_Generator/norm_deconv1/subSub,Cartoon_Generator/deconv1_2/BiasAdd:output:0,Cartoon_Generator/norm_deconv1/Mean:output:0*
T0*2
_output_shapes 
:�������������
&Cartoon_Generator/norm_deconv1/truedivRealDiv&Cartoon_Generator/norm_deconv1/sub:z:0&Cartoon_Generator/norm_deconv1/add:z:0*2
_output_shapes 
:������������*
T0�
5Cartoon_Generator/norm_deconv1/Reshape/ReadVariableOpReadVariableOpHcartoon_generator_norm_deconv1_reshape_readvariableop_norm_deconv1_gamma*
dtype0*
_output_shapes
:�
,Cartoon_Generator/norm_deconv1/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            �
&Cartoon_Generator/norm_deconv1/ReshapeReshape=Cartoon_Generator/norm_deconv1/Reshape/ReadVariableOp:value:05Cartoon_Generator/norm_deconv1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
"Cartoon_Generator/norm_deconv1/mulMul*Cartoon_Generator/norm_deconv1/truediv:z:0/Cartoon_Generator/norm_deconv1/Reshape:output:0*2
_output_shapes 
:������������*
T0�
7Cartoon_Generator/norm_deconv1/Reshape_1/ReadVariableOpReadVariableOpIcartoon_generator_norm_deconv1_reshape_1_readvariableop_norm_deconv1_beta*
dtype0*
_output_shapes
:�
.Cartoon_Generator/norm_deconv1/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
(Cartoon_Generator/norm_deconv1/Reshape_1Reshape?Cartoon_Generator/norm_deconv1/Reshape_1/ReadVariableOp:value:07Cartoon_Generator/norm_deconv1/Reshape_1/shape:output:0*&
_output_shapes
:*
T0�
$Cartoon_Generator/norm_deconv1/add_1Add&Cartoon_Generator/norm_deconv1/mul:z:01Cartoon_Generator/norm_deconv1/Reshape_1:output:0*
T0*2
_output_shapes 
:�������������
$Cartoon_Generator/activation_11/ReluRelu(Cartoon_Generator/norm_deconv1/add_1:z:0*
T0*2
_output_shapes 
:�������������
!Cartoon_Generator/deconv2_1/ShapeShape2Cartoon_Generator/activation_11/Relu:activations:0*
T0*
_output_shapes
:y
/Cartoon_Generator/deconv2_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:{
1Cartoon_Generator/deconv2_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:{
1Cartoon_Generator/deconv2_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
)Cartoon_Generator/deconv2_1/strided_sliceStridedSlice*Cartoon_Generator/deconv2_1/Shape:output:08Cartoon_Generator/deconv2_1/strided_slice/stack:output:0:Cartoon_Generator/deconv2_1/strided_slice/stack_1:output:0:Cartoon_Generator/deconv2_1/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0{
1Cartoon_Generator/deconv2_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:}
3Cartoon_Generator/deconv2_1/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0}
3Cartoon_Generator/deconv2_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
+Cartoon_Generator/deconv2_1/strided_slice_1StridedSlice*Cartoon_Generator/deconv2_1/Shape:output:0:Cartoon_Generator/deconv2_1/strided_slice_1/stack:output:0<Cartoon_Generator/deconv2_1/strided_slice_1/stack_1:output:0<Cartoon_Generator/deconv2_1/strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask{
1Cartoon_Generator/deconv2_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:}
3Cartoon_Generator/deconv2_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:}
3Cartoon_Generator/deconv2_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
+Cartoon_Generator/deconv2_1/strided_slice_2StridedSlice*Cartoon_Generator/deconv2_1/Shape:output:0:Cartoon_Generator/deconv2_1/strided_slice_2/stack:output:0<Cartoon_Generator/deconv2_1/strided_slice_2/stack_1:output:0<Cartoon_Generator/deconv2_1/strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: c
!Cartoon_Generator/deconv2_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: �
Cartoon_Generator/deconv2_1/mulMul4Cartoon_Generator/deconv2_1/strided_slice_1:output:0*Cartoon_Generator/deconv2_1/mul/y:output:0*
T0*
_output_shapes
: e
#Cartoon_Generator/deconv2_1/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :�
!Cartoon_Generator/deconv2_1/mul_1Mul4Cartoon_Generator/deconv2_1/strided_slice_2:output:0,Cartoon_Generator/deconv2_1/mul_1/y:output:0*
T0*
_output_shapes
: f
#Cartoon_Generator/deconv2_1/stack/3Const*
_output_shapes
: *
value
B :�*
dtype0�
!Cartoon_Generator/deconv2_1/stackPack2Cartoon_Generator/deconv2_1/strided_slice:output:0#Cartoon_Generator/deconv2_1/mul:z:0%Cartoon_Generator/deconv2_1/mul_1:z:0,Cartoon_Generator/deconv2_1/stack/3:output:0*
T0*
N*
_output_shapes
:{
1Cartoon_Generator/deconv2_1/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:}
3Cartoon_Generator/deconv2_1/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:}
3Cartoon_Generator/deconv2_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
+Cartoon_Generator/deconv2_1/strided_slice_3StridedSlice*Cartoon_Generator/deconv2_1/stack:output:0:Cartoon_Generator/deconv2_1/strided_slice_3/stack:output:0<Cartoon_Generator/deconv2_1/strided_slice_3/stack_1:output:0<Cartoon_Generator/deconv2_1/strided_slice_3/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask�
;Cartoon_Generator/deconv2_1/conv2d_transpose/ReadVariableOpReadVariableOpLcartoon_generator_deconv2_1_conv2d_transpose_readvariableop_deconv2_1_kernel*
dtype0*(
_output_shapes
:���
,Cartoon_Generator/deconv2_1/conv2d_transposeConv2DBackpropInput*Cartoon_Generator/deconv2_1/stack:output:0CCartoon_Generator/deconv2_1/conv2d_transpose/ReadVariableOp:value:02Cartoon_Generator/activation_11/Relu:activations:0*
paddingSAME*2
_output_shapes 
:������������*
T0*
strides
�
2Cartoon_Generator/deconv2_1/BiasAdd/ReadVariableOpReadVariableOpAcartoon_generator_deconv2_1_biasadd_readvariableop_deconv2_1_bias*
_output_shapes	
:�*
dtype0�
#Cartoon_Generator/deconv2_1/BiasAddBiasAdd5Cartoon_Generator/deconv2_1/conv2d_transpose:output:0:Cartoon_Generator/deconv2_1/BiasAdd/ReadVariableOp:value:0*2
_output_shapes 
:������������*
T0�
1Cartoon_Generator/deconv2_2/Conv2D/ReadVariableOpReadVariableOpBcartoon_generator_deconv2_2_conv2d_readvariableop_deconv2_2_kernel*
dtype0*(
_output_shapes
:���
"Cartoon_Generator/deconv2_2/Conv2DConv2D,Cartoon_Generator/deconv2_1/BiasAdd:output:09Cartoon_Generator/deconv2_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*2
_output_shapes 
:�������������
2Cartoon_Generator/deconv2_2/BiasAdd/ReadVariableOpReadVariableOpAcartoon_generator_deconv2_2_biasadd_readvariableop_deconv2_2_bias*
dtype0*
_output_shapes	
:��
#Cartoon_Generator/deconv2_2/BiasAddBiasAdd+Cartoon_Generator/deconv2_2/Conv2D:output:0:Cartoon_Generator/deconv2_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:�������������
5Cartoon_Generator/norm_deconv2/Mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0�
#Cartoon_Generator/norm_deconv2/MeanMean,Cartoon_Generator/deconv2_2/BiasAdd:output:0>Cartoon_Generator/norm_deconv2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
PCartoon_Generator/norm_deconv2/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*!
valueB"         *
dtype0�
>Cartoon_Generator/norm_deconv2/reduce_std/reduce_variance/MeanMean,Cartoon_Generator/deconv2_2/BiasAdd:output:0YCartoon_Generator/norm_deconv2/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
=Cartoon_Generator/norm_deconv2/reduce_std/reduce_variance/subSub,Cartoon_Generator/deconv2_2/BiasAdd:output:0GCartoon_Generator/norm_deconv2/reduce_std/reduce_variance/Mean:output:0*
T0*2
_output_shapes 
:�������������
@Cartoon_Generator/norm_deconv2/reduce_std/reduce_variance/SquareSquareACartoon_Generator/norm_deconv2/reduce_std/reduce_variance/sub:z:0*
T0*2
_output_shapes 
:�������������
RCartoon_Generator/norm_deconv2/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*!
valueB"         *
dtype0*
_output_shapes
:�
@Cartoon_Generator/norm_deconv2/reduce_std/reduce_variance/Mean_1MeanDCartoon_Generator/norm_deconv2/reduce_std/reduce_variance/Square:y:0[Cartoon_Generator/norm_deconv2/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
.Cartoon_Generator/norm_deconv2/reduce_std/SqrtSqrtICartoon_Generator/norm_deconv2/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:���������i
$Cartoon_Generator/norm_deconv2/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
"Cartoon_Generator/norm_deconv2/addAdd2Cartoon_Generator/norm_deconv2/reduce_std/Sqrt:y:0-Cartoon_Generator/norm_deconv2/add/y:output:0*
T0*/
_output_shapes
:����������
"Cartoon_Generator/norm_deconv2/subSub,Cartoon_Generator/deconv2_2/BiasAdd:output:0,Cartoon_Generator/norm_deconv2/Mean:output:0*
T0*2
_output_shapes 
:�������������
&Cartoon_Generator/norm_deconv2/truedivRealDiv&Cartoon_Generator/norm_deconv2/sub:z:0&Cartoon_Generator/norm_deconv2/add:z:0*2
_output_shapes 
:������������*
T0�
5Cartoon_Generator/norm_deconv2/Reshape/ReadVariableOpReadVariableOpHcartoon_generator_norm_deconv2_reshape_readvariableop_norm_deconv2_gamma*
dtype0*
_output_shapes
:�
,Cartoon_Generator/norm_deconv2/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
&Cartoon_Generator/norm_deconv2/ReshapeReshape=Cartoon_Generator/norm_deconv2/Reshape/ReadVariableOp:value:05Cartoon_Generator/norm_deconv2/Reshape/shape:output:0*&
_output_shapes
:*
T0�
"Cartoon_Generator/norm_deconv2/mulMul*Cartoon_Generator/norm_deconv2/truediv:z:0/Cartoon_Generator/norm_deconv2/Reshape:output:0*
T0*2
_output_shapes 
:�������������
7Cartoon_Generator/norm_deconv2/Reshape_1/ReadVariableOpReadVariableOpIcartoon_generator_norm_deconv2_reshape_1_readvariableop_norm_deconv2_beta*
dtype0*
_output_shapes
:�
.Cartoon_Generator/norm_deconv2/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:�
(Cartoon_Generator/norm_deconv2/Reshape_1Reshape?Cartoon_Generator/norm_deconv2/Reshape_1/ReadVariableOp:value:07Cartoon_Generator/norm_deconv2/Reshape_1/shape:output:0*
T0*&
_output_shapes
:�
$Cartoon_Generator/norm_deconv2/add_1Add&Cartoon_Generator/norm_deconv2/mul:z:01Cartoon_Generator/norm_deconv2/Reshape_1:output:0*
T0*2
_output_shapes 
:�������������
$Cartoon_Generator/activation_12/ReluRelu(Cartoon_Generator/norm_deconv2/add_1:z:0*
T0*2
_output_shapes 
:�������������
0Cartoon_Generator/zero_padding2d_17/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:�
'Cartoon_Generator/zero_padding2d_17/PadPad2Cartoon_Generator/activation_12/Relu:activations:09Cartoon_Generator/zero_padding2d_17/Pad/paddings:output:0*
T0*2
_output_shapes 
:�������������
/Cartoon_Generator/deconv3/Conv2D/ReadVariableOpReadVariableOp>cartoon_generator_deconv3_conv2d_readvariableop_deconv3_kernel*
dtype0*'
_output_shapes
:��
 Cartoon_Generator/deconv3/Conv2DConv2D0Cartoon_Generator/zero_padding2d_17/Pad:output:07Cartoon_Generator/deconv3/Conv2D/ReadVariableOp:value:0*
paddingVALID*1
_output_shapes
:�����������*
T0*
strides
�
0Cartoon_Generator/deconv3/BiasAdd/ReadVariableOpReadVariableOp=cartoon_generator_deconv3_biasadd_readvariableop_deconv3_bias*
dtype0*
_output_shapes
:�
!Cartoon_Generator/deconv3/BiasAddBiasAdd)Cartoon_Generator/deconv3/Conv2D:output:08Cartoon_Generator/deconv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
$Cartoon_Generator/activation_13/TanhTanh*Cartoon_Generator/deconv3/BiasAdd:output:0*1
_output_shapes
:�����������*
T0�&
IdentityIdentity(Cartoon_Generator/activation_13/Tanh:y:03^Cartoon_Generator/norm7_2/Reshape_1/ReadVariableOp1^Cartoon_Generator/conv10_2/Conv2D/ReadVariableOp8^Cartoon_Generator/norm_deconv2/Reshape_1/ReadVariableOp2^Cartoon_Generator/norm10_1/Reshape/ReadVariableOp3^Cartoon_Generator/norm8_1/Reshape_1/ReadVariableOp2^Cartoon_Generator/deconv2_2/Conv2D/ReadVariableOp0^Cartoon_Generator/conv3_1/Conv2D/ReadVariableOp3^Cartoon_Generator/deconv2_1/BiasAdd/ReadVariableOp1^Cartoon_Generator/norm8_1/Reshape/ReadVariableOp1^Cartoon_Generator/conv9_1/BiasAdd/ReadVariableOp0^Cartoon_Generator/conv3_2/Conv2D/ReadVariableOp3^Cartoon_Generator/norm9_2/Reshape_1/ReadVariableOp1^Cartoon_Generator/conv3_2/BiasAdd/ReadVariableOp0^Cartoon_Generator/conv9_1/Conv2D/ReadVariableOp1^Cartoon_Generator/norm2/Reshape_1/ReadVariableOp2^Cartoon_Generator/conv11_1/BiasAdd/ReadVariableOp2^Cartoon_Generator/conv10_2/BiasAdd/ReadVariableOp3^Cartoon_Generator/deconv1_1/BiasAdd/ReadVariableOp1^Cartoon_Generator/norm4_2/Reshape/ReadVariableOp<^Cartoon_Generator/deconv1_1/conv2d_transpose/ReadVariableOp3^Cartoon_Generator/deconv2_2/BiasAdd/ReadVariableOp8^Cartoon_Generator/norm_deconv1/Reshape_1/ReadVariableOp1^Cartoon_Generator/conv2_2/BiasAdd/ReadVariableOp.^Cartoon_Generator/conv1/Conv2D/ReadVariableOp/^Cartoon_Generator/norm3/Reshape/ReadVariableOp0^Cartoon_Generator/conv7_2/Conv2D/ReadVariableOp1^Cartoon_Generator/conv2_1/BiasAdd/ReadVariableOp0^Cartoon_Generator/conv5_2/Conv2D/ReadVariableOp2^Cartoon_Generator/deconv1_2/Conv2D/ReadVariableOp2^Cartoon_Generator/conv11_2/BiasAdd/ReadVariableOp3^Cartoon_Generator/norm4_2/Reshape_1/ReadVariableOp3^Cartoon_Generator/norm7_1/Reshape_1/ReadVariableOp2^Cartoon_Generator/conv10_1/BiasAdd/ReadVariableOp1^Cartoon_Generator/conv11_1/Conv2D/ReadVariableOp1^Cartoon_Generator/conv3_1/BiasAdd/ReadVariableOp1^Cartoon_Generator/deconv3/BiasAdd/ReadVariableOp1^Cartoon_Generator/norm7_2/Reshape/ReadVariableOp3^Cartoon_Generator/norm6_2/Reshape_1/ReadVariableOp/^Cartoon_Generator/conv1/BiasAdd/ReadVariableOp1^Cartoon_Generator/norm4_1/Reshape/ReadVariableOp1^Cartoon_Generator/conv11_2/Conv2D/ReadVariableOp1^Cartoon_Generator/conv10_1/Conv2D/ReadVariableOp4^Cartoon_Generator/norm11_2/Reshape_1/ReadVariableOp0^Cartoon_Generator/conv9_2/Conv2D/ReadVariableOp0^Cartoon_Generator/conv8_1/Conv2D/ReadVariableOp0^Cartoon_Generator/deconv3/Conv2D/ReadVariableOp1^Cartoon_Generator/conv8_2/BiasAdd/ReadVariableOp4^Cartoon_Generator/norm10_2/Reshape_1/ReadVariableOp1^Cartoon_Generator/conv7_1/BiasAdd/ReadVariableOp2^Cartoon_Generator/norm10_2/Reshape/ReadVariableOp1^Cartoon_Generator/conv6_2/BiasAdd/ReadVariableOp0^Cartoon_Generator/conv6_2/Conv2D/ReadVariableOp1^Cartoon_Generator/norm5_2/Reshape/ReadVariableOp2^Cartoon_Generator/norm11_1/Reshape/ReadVariableOp1^Cartoon_Generator/conv4_1/BiasAdd/ReadVariableOp0^Cartoon_Generator/conv4_2/Conv2D/ReadVariableOp3^Cartoon_Generator/norm8_2/Reshape_1/ReadVariableOp1^Cartoon_Generator/conv4_2/BiasAdd/ReadVariableOp1^Cartoon_Generator/conv5_2/BiasAdd/ReadVariableOp6^Cartoon_Generator/norm_deconv2/Reshape/ReadVariableOp3^Cartoon_Generator/norm9_1/Reshape_1/ReadVariableOp3^Cartoon_Generator/norm4_1/Reshape_1/ReadVariableOp1^Cartoon_Generator/norm6_1/Reshape/ReadVariableOp1^Cartoon_Generator/conv8_1/BiasAdd/ReadVariableOp1^Cartoon_Generator/norm9_1/Reshape/ReadVariableOp1^Cartoon_Generator/norm8_2/Reshape/ReadVariableOp0^Cartoon_Generator/conv2_2/Conv2D/ReadVariableOp/^Cartoon_Generator/norm1/Reshape/ReadVariableOp0^Cartoon_Generator/conv5_1/Conv2D/ReadVariableOp1^Cartoon_Generator/norm1/Reshape_1/ReadVariableOp3^Cartoon_Generator/norm6_1/Reshape_1/ReadVariableOp3^Cartoon_Generator/deconv1_2/BiasAdd/ReadVariableOp1^Cartoon_Generator/norm5_1/Reshape/ReadVariableOp0^Cartoon_Generator/conv6_1/Conv2D/ReadVariableOp3^Cartoon_Generator/norm5_2/Reshape_1/ReadVariableOp3^Cartoon_Generator/norm5_1/Reshape_1/ReadVariableOp1^Cartoon_Generator/conv6_1/BiasAdd/ReadVariableOp1^Cartoon_Generator/norm9_2/Reshape/ReadVariableOp0^Cartoon_Generator/conv2_1/Conv2D/ReadVariableOp<^Cartoon_Generator/deconv2_1/conv2d_transpose/ReadVariableOp2^Cartoon_Generator/norm11_2/Reshape/ReadVariableOp1^Cartoon_Generator/conv9_2/BiasAdd/ReadVariableOp4^Cartoon_Generator/norm10_1/Reshape_1/ReadVariableOp1^Cartoon_Generator/norm7_1/Reshape/ReadVariableOp0^Cartoon_Generator/conv8_2/Conv2D/ReadVariableOp0^Cartoon_Generator/conv4_1/Conv2D/ReadVariableOp6^Cartoon_Generator/norm_deconv1/Reshape/ReadVariableOp1^Cartoon_Generator/norm6_2/Reshape/ReadVariableOp/^Cartoon_Generator/norm2/Reshape/ReadVariableOp1^Cartoon_Generator/conv7_2/BiasAdd/ReadVariableOp1^Cartoon_Generator/conv5_1/BiasAdd/ReadVariableOp4^Cartoon_Generator/norm11_1/Reshape_1/ReadVariableOp1^Cartoon_Generator/norm3/Reshape_1/ReadVariableOp0^Cartoon_Generator/conv7_1/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2f
1Cartoon_Generator/conv10_1/BiasAdd/ReadVariableOp1Cartoon_Generator/conv10_1/BiasAdd/ReadVariableOp2h
2Cartoon_Generator/norm8_2/Reshape_1/ReadVariableOp2Cartoon_Generator/norm8_2/Reshape_1/ReadVariableOp2j
3Cartoon_Generator/norm10_1/Reshape_1/ReadVariableOp3Cartoon_Generator/norm10_1/Reshape_1/ReadVariableOp2h
2Cartoon_Generator/norm5_1/Reshape_1/ReadVariableOp2Cartoon_Generator/norm5_1/Reshape_1/ReadVariableOp2h
2Cartoon_Generator/deconv2_1/BiasAdd/ReadVariableOp2Cartoon_Generator/deconv2_1/BiasAdd/ReadVariableOp2b
/Cartoon_Generator/conv3_1/Conv2D/ReadVariableOp/Cartoon_Generator/conv3_1/Conv2D/ReadVariableOp2d
0Cartoon_Generator/norm6_1/Reshape/ReadVariableOp0Cartoon_Generator/norm6_1/Reshape/ReadVariableOp2^
-Cartoon_Generator/conv1/Conv2D/ReadVariableOp-Cartoon_Generator/conv1/Conv2D/ReadVariableOp2b
/Cartoon_Generator/conv7_1/Conv2D/ReadVariableOp/Cartoon_Generator/conv7_1/Conv2D/ReadVariableOp2d
0Cartoon_Generator/conv2_2/BiasAdd/ReadVariableOp0Cartoon_Generator/conv2_2/BiasAdd/ReadVariableOp2d
0Cartoon_Generator/conv5_1/BiasAdd/ReadVariableOp0Cartoon_Generator/conv5_1/BiasAdd/ReadVariableOp2`
.Cartoon_Generator/norm3/Reshape/ReadVariableOp.Cartoon_Generator/norm3/Reshape/ReadVariableOp2d
0Cartoon_Generator/conv9_2/BiasAdd/ReadVariableOp0Cartoon_Generator/conv9_2/BiasAdd/ReadVariableOp2h
2Cartoon_Generator/norm4_2/Reshape_1/ReadVariableOp2Cartoon_Generator/norm4_2/Reshape_1/ReadVariableOp2b
/Cartoon_Generator/conv5_2/Conv2D/ReadVariableOp/Cartoon_Generator/conv5_2/Conv2D/ReadVariableOp2f
1Cartoon_Generator/conv11_2/BiasAdd/ReadVariableOp1Cartoon_Generator/conv11_2/BiasAdd/ReadVariableOp2b
/Cartoon_Generator/conv9_2/Conv2D/ReadVariableOp/Cartoon_Generator/conv9_2/Conv2D/ReadVariableOp2d
0Cartoon_Generator/conv11_2/Conv2D/ReadVariableOp0Cartoon_Generator/conv11_2/Conv2D/ReadVariableOp2d
0Cartoon_Generator/norm7_2/Reshape/ReadVariableOp0Cartoon_Generator/norm7_2/Reshape/ReadVariableOp2d
0Cartoon_Generator/conv2_1/BiasAdd/ReadVariableOp0Cartoon_Generator/conv2_1/BiasAdd/ReadVariableOp2d
0Cartoon_Generator/conv6_2/BiasAdd/ReadVariableOp0Cartoon_Generator/conv6_2/BiasAdd/ReadVariableOp2j
3Cartoon_Generator/norm11_1/Reshape_1/ReadVariableOp3Cartoon_Generator/norm11_1/Reshape_1/ReadVariableOp2`
.Cartoon_Generator/norm2/Reshape/ReadVariableOp.Cartoon_Generator/norm2/Reshape/ReadVariableOp2h
2Cartoon_Generator/norm9_2/Reshape_1/ReadVariableOp2Cartoon_Generator/norm9_2/Reshape_1/ReadVariableOp2d
0Cartoon_Generator/norm3/Reshape_1/ReadVariableOp0Cartoon_Generator/norm3/Reshape_1/ReadVariableOp2d
0Cartoon_Generator/conv9_1/BiasAdd/ReadVariableOp0Cartoon_Generator/conv9_1/BiasAdd/ReadVariableOp2f
1Cartoon_Generator/deconv1_2/Conv2D/ReadVariableOp1Cartoon_Generator/deconv1_2/Conv2D/ReadVariableOp2h
2Cartoon_Generator/norm6_1/Reshape_1/ReadVariableOp2Cartoon_Generator/norm6_1/Reshape_1/ReadVariableOp2f
1Cartoon_Generator/conv11_1/BiasAdd/ReadVariableOp1Cartoon_Generator/conv11_1/BiasAdd/ReadVariableOp2z
;Cartoon_Generator/deconv1_1/conv2d_transpose/ReadVariableOp;Cartoon_Generator/deconv1_1/conv2d_transpose/ReadVariableOp2b
/Cartoon_Generator/conv4_1/Conv2D/ReadVariableOp/Cartoon_Generator/conv4_1/Conv2D/ReadVariableOp2j
3Cartoon_Generator/norm10_2/Reshape_1/ReadVariableOp3Cartoon_Generator/norm10_2/Reshape_1/ReadVariableOp2d
0Cartoon_Generator/norm4_2/Reshape/ReadVariableOp0Cartoon_Generator/norm4_2/Reshape/ReadVariableOp2f
1Cartoon_Generator/norm10_2/Reshape/ReadVariableOp1Cartoon_Generator/norm10_2/Reshape/ReadVariableOp2b
/Cartoon_Generator/conv8_1/Conv2D/ReadVariableOp/Cartoon_Generator/conv8_1/Conv2D/ReadVariableOp2h
2Cartoon_Generator/norm5_2/Reshape_1/ReadVariableOp2Cartoon_Generator/norm5_2/Reshape_1/ReadVariableOp2d
0Cartoon_Generator/conv10_1/Conv2D/ReadVariableOp0Cartoon_Generator/conv10_1/Conv2D/ReadVariableOp2d
0Cartoon_Generator/norm7_1/Reshape/ReadVariableOp0Cartoon_Generator/norm7_1/Reshape/ReadVariableOp2b
/Cartoon_Generator/conv2_2/Conv2D/ReadVariableOp/Cartoon_Generator/conv2_2/Conv2D/ReadVariableOp2d
0Cartoon_Generator/conv3_2/BiasAdd/ReadVariableOp0Cartoon_Generator/conv3_2/BiasAdd/ReadVariableOp2d
0Cartoon_Generator/conv6_1/BiasAdd/ReadVariableOp0Cartoon_Generator/conv6_1/BiasAdd/ReadVariableOp2`
.Cartoon_Generator/norm1/Reshape/ReadVariableOp.Cartoon_Generator/norm1/Reshape/ReadVariableOp2b
/Cartoon_Generator/conv6_2/Conv2D/ReadVariableOp/Cartoon_Generator/conv6_2/Conv2D/ReadVariableOp2d
0Cartoon_Generator/norm4_1/Reshape/ReadVariableOp0Cartoon_Generator/norm4_1/Reshape/ReadVariableOp2f
1Cartoon_Generator/norm10_1/Reshape/ReadVariableOp1Cartoon_Generator/norm10_1/Reshape/ReadVariableOp2d
0Cartoon_Generator/norm8_2/Reshape/ReadVariableOp0Cartoon_Generator/norm8_2/Reshape/ReadVariableOp2h
2Cartoon_Generator/norm7_1/Reshape_1/ReadVariableOp2Cartoon_Generator/norm7_1/Reshape_1/ReadVariableOp2z
;Cartoon_Generator/deconv2_1/conv2d_transpose/ReadVariableOp;Cartoon_Generator/deconv2_1/conv2d_transpose/ReadVariableOp2d
0Cartoon_Generator/conv3_1/BiasAdd/ReadVariableOp0Cartoon_Generator/conv3_1/BiasAdd/ReadVariableOp2f
1Cartoon_Generator/deconv2_2/Conv2D/ReadVariableOp1Cartoon_Generator/deconv2_2/Conv2D/ReadVariableOp2d
0Cartoon_Generator/conv7_2/BiasAdd/ReadVariableOp0Cartoon_Generator/conv7_2/BiasAdd/ReadVariableOp2j
3Cartoon_Generator/norm11_2/Reshape_1/ReadVariableOp3Cartoon_Generator/norm11_2/Reshape_1/ReadVariableOp2h
2Cartoon_Generator/norm6_2/Reshape_1/ReadVariableOp2Cartoon_Generator/norm6_2/Reshape_1/ReadVariableOp2b
/Cartoon_Generator/conv5_1/Conv2D/ReadVariableOp/Cartoon_Generator/conv5_1/Conv2D/ReadVariableOp2`
.Cartoon_Generator/conv1/BiasAdd/ReadVariableOp.Cartoon_Generator/conv1/BiasAdd/ReadVariableOp2b
/Cartoon_Generator/conv9_1/Conv2D/ReadVariableOp/Cartoon_Generator/conv9_1/Conv2D/ReadVariableOp2n
5Cartoon_Generator/norm_deconv2/Reshape/ReadVariableOp5Cartoon_Generator/norm_deconv2/Reshape/ReadVariableOp2d
0Cartoon_Generator/conv11_1/Conv2D/ReadVariableOp0Cartoon_Generator/conv11_1/Conv2D/ReadVariableOp2h
2Cartoon_Generator/deconv1_2/BiasAdd/ReadVariableOp2Cartoon_Generator/deconv1_2/BiasAdd/ReadVariableOp2d
0Cartoon_Generator/norm5_2/Reshape/ReadVariableOp0Cartoon_Generator/norm5_2/Reshape/ReadVariableOp2b
/Cartoon_Generator/conv3_2/Conv2D/ReadVariableOp/Cartoon_Generator/conv3_2/Conv2D/ReadVariableOp2f
1Cartoon_Generator/norm11_2/Reshape/ReadVariableOp1Cartoon_Generator/norm11_2/Reshape/ReadVariableOp2d
0Cartoon_Generator/norm8_1/Reshape/ReadVariableOp0Cartoon_Generator/norm8_1/Reshape/ReadVariableOp2b
/Cartoon_Generator/conv7_2/Conv2D/ReadVariableOp/Cartoon_Generator/conv7_2/Conv2D/ReadVariableOp2d
0Cartoon_Generator/conv4_2/BiasAdd/ReadVariableOp0Cartoon_Generator/conv4_2/BiasAdd/ReadVariableOp2r
7Cartoon_Generator/norm_deconv1/Reshape_1/ReadVariableOp7Cartoon_Generator/norm_deconv1/Reshape_1/ReadVariableOp2d
0Cartoon_Generator/conv7_1/BiasAdd/ReadVariableOp0Cartoon_Generator/conv7_1/BiasAdd/ReadVariableOp2d
0Cartoon_Generator/norm1/Reshape_1/ReadVariableOp0Cartoon_Generator/norm1/Reshape_1/ReadVariableOp2h
2Cartoon_Generator/norm8_1/Reshape_1/ReadVariableOp2Cartoon_Generator/norm8_1/Reshape_1/ReadVariableOp2n
5Cartoon_Generator/norm_deconv1/Reshape/ReadVariableOp5Cartoon_Generator/norm_deconv1/Reshape/ReadVariableOp2h
2Cartoon_Generator/deconv1_1/BiasAdd/ReadVariableOp2Cartoon_Generator/deconv1_1/BiasAdd/ReadVariableOp2d
0Cartoon_Generator/norm5_1/Reshape/ReadVariableOp0Cartoon_Generator/norm5_1/Reshape/ReadVariableOp2f
1Cartoon_Generator/norm11_1/Reshape/ReadVariableOp1Cartoon_Generator/norm11_1/Reshape/ReadVariableOp2d
0Cartoon_Generator/norm9_2/Reshape/ReadVariableOp0Cartoon_Generator/norm9_2/Reshape/ReadVariableOp2h
2Cartoon_Generator/norm7_2/Reshape_1/ReadVariableOp2Cartoon_Generator/norm7_2/Reshape_1/ReadVariableOp2b
/Cartoon_Generator/conv2_1/Conv2D/ReadVariableOp/Cartoon_Generator/conv2_1/Conv2D/ReadVariableOp2d
0Cartoon_Generator/conv4_1/BiasAdd/ReadVariableOp0Cartoon_Generator/conv4_1/BiasAdd/ReadVariableOp2d
0Cartoon_Generator/conv8_2/BiasAdd/ReadVariableOp0Cartoon_Generator/conv8_2/BiasAdd/ReadVariableOp2b
/Cartoon_Generator/conv6_1/Conv2D/ReadVariableOp/Cartoon_Generator/conv6_1/Conv2D/ReadVariableOp2h
2Cartoon_Generator/norm4_1/Reshape_1/ReadVariableOp2Cartoon_Generator/norm4_1/Reshape_1/ReadVariableOp2f
1Cartoon_Generator/conv10_2/BiasAdd/ReadVariableOp1Cartoon_Generator/conv10_2/BiasAdd/ReadVariableOp2b
/Cartoon_Generator/conv4_2/Conv2D/ReadVariableOp/Cartoon_Generator/conv4_2/Conv2D/ReadVariableOp2h
2Cartoon_Generator/deconv2_2/BiasAdd/ReadVariableOp2Cartoon_Generator/deconv2_2/BiasAdd/ReadVariableOp2d
0Cartoon_Generator/norm6_2/Reshape/ReadVariableOp0Cartoon_Generator/norm6_2/Reshape/ReadVariableOp2b
/Cartoon_Generator/conv8_2/Conv2D/ReadVariableOp/Cartoon_Generator/conv8_2/Conv2D/ReadVariableOp2d
0Cartoon_Generator/conv10_2/Conv2D/ReadVariableOp0Cartoon_Generator/conv10_2/Conv2D/ReadVariableOp2d
0Cartoon_Generator/norm9_1/Reshape/ReadVariableOp0Cartoon_Generator/norm9_1/Reshape/ReadVariableOp2d
0Cartoon_Generator/deconv3/BiasAdd/ReadVariableOp0Cartoon_Generator/deconv3/BiasAdd/ReadVariableOp2b
/Cartoon_Generator/deconv3/Conv2D/ReadVariableOp/Cartoon_Generator/deconv3/Conv2D/ReadVariableOp2r
7Cartoon_Generator/norm_deconv2/Reshape_1/ReadVariableOp7Cartoon_Generator/norm_deconv2/Reshape_1/ReadVariableOp2d
0Cartoon_Generator/conv5_2/BiasAdd/ReadVariableOp0Cartoon_Generator/conv5_2/BiasAdd/ReadVariableOp2h
2Cartoon_Generator/norm9_1/Reshape_1/ReadVariableOp2Cartoon_Generator/norm9_1/Reshape_1/ReadVariableOp2d
0Cartoon_Generator/norm2/Reshape_1/ReadVariableOp0Cartoon_Generator/norm2/Reshape_1/ReadVariableOp2d
0Cartoon_Generator/conv8_1/BiasAdd/ReadVariableOp0Cartoon_Generator/conv8_1/BiasAdd/ReadVariableOp: : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :H :I :J :K :L :M :N :O :P :Q :R :S :T :U :V :W :X :Y :Z :[ :\ :] :^ :% !

_user_specified_nameinput: : : : : : : : :	 :
 : : : : : : : : : : : : 
��
�#
__inference__traced_save_7584
file_prefix-
)savev2_conv1_1_kernel_read_readvariableop+
'savev2_conv1_1_bias_read_readvariableop*
&savev2_norm1_gamma_read_readvariableop)
%savev2_norm1_beta_read_readvariableop/
+savev2_conv2_1_1_kernel_read_readvariableop-
)savev2_conv2_1_1_bias_read_readvariableop/
+savev2_conv2_2_1_kernel_read_readvariableop-
)savev2_conv2_2_1_bias_read_readvariableop*
&savev2_norm2_gamma_read_readvariableop)
%savev2_norm2_beta_read_readvariableop/
+savev2_conv3_1_1_kernel_read_readvariableop-
)savev2_conv3_1_1_bias_read_readvariableop/
+savev2_conv3_2_1_kernel_read_readvariableop-
)savev2_conv3_2_1_bias_read_readvariableop*
&savev2_norm3_gamma_read_readvariableop)
%savev2_norm3_beta_read_readvariableop-
)savev2_conv4_1_kernel_read_readvariableop+
'savev2_conv4_1_bias_read_readvariableop,
(savev2_norm4_1_gamma_read_readvariableop+
'savev2_norm4_1_beta_read_readvariableop-
)savev2_conv4_2_kernel_read_readvariableop+
'savev2_conv4_2_bias_read_readvariableop,
(savev2_norm4_2_gamma_read_readvariableop+
'savev2_norm4_2_beta_read_readvariableop-
)savev2_conv5_1_kernel_read_readvariableop+
'savev2_conv5_1_bias_read_readvariableop,
(savev2_norm5_1_gamma_read_readvariableop+
'savev2_norm5_1_beta_read_readvariableop-
)savev2_conv5_2_kernel_read_readvariableop+
'savev2_conv5_2_bias_read_readvariableop,
(savev2_norm5_2_gamma_read_readvariableop+
'savev2_norm5_2_beta_read_readvariableop-
)savev2_conv6_1_kernel_read_readvariableop+
'savev2_conv6_1_bias_read_readvariableop,
(savev2_norm6_1_gamma_read_readvariableop+
'savev2_norm6_1_beta_read_readvariableop-
)savev2_conv6_2_kernel_read_readvariableop+
'savev2_conv6_2_bias_read_readvariableop,
(savev2_norm6_2_gamma_read_readvariableop+
'savev2_norm6_2_beta_read_readvariableop-
)savev2_conv7_1_kernel_read_readvariableop+
'savev2_conv7_1_bias_read_readvariableop,
(savev2_norm7_1_gamma_read_readvariableop+
'savev2_norm7_1_beta_read_readvariableop-
)savev2_conv7_2_kernel_read_readvariableop+
'savev2_conv7_2_bias_read_readvariableop,
(savev2_norm7_2_gamma_read_readvariableop+
'savev2_norm7_2_beta_read_readvariableop-
)savev2_conv8_1_kernel_read_readvariableop+
'savev2_conv8_1_bias_read_readvariableop,
(savev2_norm8_1_gamma_read_readvariableop+
'savev2_norm8_1_beta_read_readvariableop-
)savev2_conv8_2_kernel_read_readvariableop+
'savev2_conv8_2_bias_read_readvariableop,
(savev2_norm8_2_gamma_read_readvariableop+
'savev2_norm8_2_beta_read_readvariableop-
)savev2_conv9_1_kernel_read_readvariableop+
'savev2_conv9_1_bias_read_readvariableop,
(savev2_norm9_1_gamma_read_readvariableop+
'savev2_norm9_1_beta_read_readvariableop-
)savev2_conv9_2_kernel_read_readvariableop+
'savev2_conv9_2_bias_read_readvariableop,
(savev2_norm9_2_gamma_read_readvariableop+
'savev2_norm9_2_beta_read_readvariableop.
*savev2_conv10_1_kernel_read_readvariableop,
(savev2_conv10_1_bias_read_readvariableop-
)savev2_norm10_1_gamma_read_readvariableop,
(savev2_norm10_1_beta_read_readvariableop.
*savev2_conv10_2_kernel_read_readvariableop,
(savev2_conv10_2_bias_read_readvariableop-
)savev2_norm10_2_gamma_read_readvariableop,
(savev2_norm10_2_beta_read_readvariableop.
*savev2_conv11_1_kernel_read_readvariableop,
(savev2_conv11_1_bias_read_readvariableop-
)savev2_norm11_1_gamma_read_readvariableop,
(savev2_norm11_1_beta_read_readvariableop.
*savev2_conv11_2_kernel_read_readvariableop,
(savev2_conv11_2_bias_read_readvariableop-
)savev2_norm11_2_gamma_read_readvariableop,
(savev2_norm11_2_beta_read_readvariableop/
+savev2_deconv1_1_kernel_read_readvariableop-
)savev2_deconv1_1_bias_read_readvariableop/
+savev2_deconv1_2_kernel_read_readvariableop-
)savev2_deconv1_2_bias_read_readvariableop1
-savev2_norm_deconv1_gamma_read_readvariableop0
,savev2_norm_deconv1_beta_read_readvariableop/
+savev2_deconv2_1_kernel_read_readvariableop-
)savev2_deconv2_1_bias_read_readvariableop/
+savev2_deconv2_2_kernel_read_readvariableop-
)savev2_deconv2_2_bias_read_readvariableop1
-savev2_norm_deconv2_gamma_read_readvariableop0
,savev2_norm_deconv2_beta_read_readvariableop-
)savev2_deconv3_kernel_read_readvariableop+
'savev2_deconv3_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2_1�SaveV2�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_354cfcad52fe4375baaba8cdffad8ff5/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:^*�(
value�(B�(^B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-31/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-33/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-35/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-37/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-39/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-39/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-40/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-41/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-41/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-42/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-42/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-43/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-43/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-45/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-45/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-46/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-46/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0�
SaveV2/shape_and_slicesConst"/device:CPU:0*�
value�B�^B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:^�!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_conv1_1_kernel_read_readvariableop'savev2_conv1_1_bias_read_readvariableop&savev2_norm1_gamma_read_readvariableop%savev2_norm1_beta_read_readvariableop+savev2_conv2_1_1_kernel_read_readvariableop)savev2_conv2_1_1_bias_read_readvariableop+savev2_conv2_2_1_kernel_read_readvariableop)savev2_conv2_2_1_bias_read_readvariableop&savev2_norm2_gamma_read_readvariableop%savev2_norm2_beta_read_readvariableop+savev2_conv3_1_1_kernel_read_readvariableop)savev2_conv3_1_1_bias_read_readvariableop+savev2_conv3_2_1_kernel_read_readvariableop)savev2_conv3_2_1_bias_read_readvariableop&savev2_norm3_gamma_read_readvariableop%savev2_norm3_beta_read_readvariableop)savev2_conv4_1_kernel_read_readvariableop'savev2_conv4_1_bias_read_readvariableop(savev2_norm4_1_gamma_read_readvariableop'savev2_norm4_1_beta_read_readvariableop)savev2_conv4_2_kernel_read_readvariableop'savev2_conv4_2_bias_read_readvariableop(savev2_norm4_2_gamma_read_readvariableop'savev2_norm4_2_beta_read_readvariableop)savev2_conv5_1_kernel_read_readvariableop'savev2_conv5_1_bias_read_readvariableop(savev2_norm5_1_gamma_read_readvariableop'savev2_norm5_1_beta_read_readvariableop)savev2_conv5_2_kernel_read_readvariableop'savev2_conv5_2_bias_read_readvariableop(savev2_norm5_2_gamma_read_readvariableop'savev2_norm5_2_beta_read_readvariableop)savev2_conv6_1_kernel_read_readvariableop'savev2_conv6_1_bias_read_readvariableop(savev2_norm6_1_gamma_read_readvariableop'savev2_norm6_1_beta_read_readvariableop)savev2_conv6_2_kernel_read_readvariableop'savev2_conv6_2_bias_read_readvariableop(savev2_norm6_2_gamma_read_readvariableop'savev2_norm6_2_beta_read_readvariableop)savev2_conv7_1_kernel_read_readvariableop'savev2_conv7_1_bias_read_readvariableop(savev2_norm7_1_gamma_read_readvariableop'savev2_norm7_1_beta_read_readvariableop)savev2_conv7_2_kernel_read_readvariableop'savev2_conv7_2_bias_read_readvariableop(savev2_norm7_2_gamma_read_readvariableop'savev2_norm7_2_beta_read_readvariableop)savev2_conv8_1_kernel_read_readvariableop'savev2_conv8_1_bias_read_readvariableop(savev2_norm8_1_gamma_read_readvariableop'savev2_norm8_1_beta_read_readvariableop)savev2_conv8_2_kernel_read_readvariableop'savev2_conv8_2_bias_read_readvariableop(savev2_norm8_2_gamma_read_readvariableop'savev2_norm8_2_beta_read_readvariableop)savev2_conv9_1_kernel_read_readvariableop'savev2_conv9_1_bias_read_readvariableop(savev2_norm9_1_gamma_read_readvariableop'savev2_norm9_1_beta_read_readvariableop)savev2_conv9_2_kernel_read_readvariableop'savev2_conv9_2_bias_read_readvariableop(savev2_norm9_2_gamma_read_readvariableop'savev2_norm9_2_beta_read_readvariableop*savev2_conv10_1_kernel_read_readvariableop(savev2_conv10_1_bias_read_readvariableop)savev2_norm10_1_gamma_read_readvariableop(savev2_norm10_1_beta_read_readvariableop*savev2_conv10_2_kernel_read_readvariableop(savev2_conv10_2_bias_read_readvariableop)savev2_norm10_2_gamma_read_readvariableop(savev2_norm10_2_beta_read_readvariableop*savev2_conv11_1_kernel_read_readvariableop(savev2_conv11_1_bias_read_readvariableop)savev2_norm11_1_gamma_read_readvariableop(savev2_norm11_1_beta_read_readvariableop*savev2_conv11_2_kernel_read_readvariableop(savev2_conv11_2_bias_read_readvariableop)savev2_norm11_2_gamma_read_readvariableop(savev2_norm11_2_beta_read_readvariableop+savev2_deconv1_1_kernel_read_readvariableop)savev2_deconv1_1_bias_read_readvariableop+savev2_deconv1_2_kernel_read_readvariableop)savev2_deconv1_2_bias_read_readvariableop-savev2_norm_deconv1_gamma_read_readvariableop,savev2_norm_deconv1_beta_read_readvariableop+savev2_deconv2_1_kernel_read_readvariableop)savev2_deconv2_1_bias_read_readvariableop+savev2_deconv2_2_kernel_read_readvariableop)savev2_deconv2_2_bias_read_readvariableop-savev2_norm_deconv2_gamma_read_readvariableop,savev2_norm_deconv2_beta_read_readvariableop)savev2_deconv3_kernel_read_readvariableop'savev2_deconv3_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *l
dtypesb
`2^h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
N�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints	^SaveV2_1^SaveV2*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:@:::@�:�:��:�:::��:�:��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:::��:�:��:�:::��:�:��:�:::�:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :H :I :J :K :L :M :N :O :P :Q :R :S :T :U :V :W :X :Y :Z :[ :\ :] :^ :_ :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 "&L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
A
input8
serving_default_input:0�����������K
activation_13:
StatefulPartitionedCall:0�����������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:�R
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer_with_weights-14
layer-26
layer_with_weights-15
layer-27
layer-28
layer-29
layer_with_weights-16
layer-30
 layer_with_weights-17
 layer-31
!layer-32
"layer-33
#layer_with_weights-18
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer_with_weights-21
(layer-39
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,layer_with_weights-23
,layer-43
-layer-44
.layer-45
/layer_with_weights-24
/layer-46
0layer_with_weights-25
0layer-47
1layer-48
2layer-49
3layer_with_weights-26
3layer-50
4layer_with_weights-27
4layer-51
5layer-52
6layer-53
7layer_with_weights-28
7layer-54
8layer_with_weights-29
8layer-55
9layer-56
:layer-57
;layer_with_weights-30
;layer-58
<layer_with_weights-31
<layer-59
=layer-60
>layer-61
?layer_with_weights-32
?layer-62
@layer_with_weights-33
@layer-63
Alayer-64
Blayer-65
Clayer_with_weights-34
Clayer-66
Dlayer_with_weights-35
Dlayer-67
Elayer-68
Flayer-69
Glayer_with_weights-36
Glayer-70
Hlayer_with_weights-37
Hlayer-71
Ilayer-72
Jlayer-73
Klayer_with_weights-38
Klayer-74
Llayer_with_weights-39
Llayer-75
Mlayer-76
Nlayer_with_weights-40
Nlayer-77
Olayer_with_weights-41
Olayer-78
Player_with_weights-42
Player-79
Qlayer-80
Rlayer_with_weights-43
Rlayer-81
Slayer_with_weights-44
Slayer-82
Tlayer_with_weights-45
Tlayer-83
Ulayer-84
Vlayer-85
Wlayer_with_weights-46
Wlayer-86
Xlayer-87
Y
signatures
�_default_save_signature"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
4

Zkernel
[bias"
_generic_user_object
3
	\gamma
]beta"
_generic_user_object
"
_generic_user_object
4

^kernel
_bias"
_generic_user_object
4

`kernel
abias"
_generic_user_object
3
	bgamma
cbeta"
_generic_user_object
"
_generic_user_object
4

dkernel
ebias"
_generic_user_object
4

fkernel
gbias"
_generic_user_object
3
	hgamma
ibeta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
4

jkernel
kbias"
_generic_user_object
3
	lgamma
mbeta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
4

nkernel
obias"
_generic_user_object
3
	pgamma
qbeta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
4

rkernel
sbias"
_generic_user_object
3
	tgamma
ubeta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
4

vkernel
wbias"
_generic_user_object
3
	xgamma
ybeta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
4

zkernel
{bias"
_generic_user_object
3
	|gamma
}beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
4

~kernel
bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
5

�gamma
	�beta"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
6
�kernel
	�bias"
_generic_user_object
"
_generic_user_object
-
�serving_default"
signature_map
(:&@2conv1_1/kernel
:@2conv1_1/bias
:2norm1/gamma
:2
norm1/beta
+:)@�2conv2_1_1/kernel
:�2conv2_1_1/bias
,:*��2conv2_2_1/kernel
:�2conv2_2_1/bias
:2norm2/gamma
:2
norm2/beta
,:*��2conv3_1_1/kernel
:�2conv3_1_1/bias
,:*��2conv3_2_1/kernel
:�2conv3_2_1/bias
:2norm3/gamma
:2
norm3/beta
*:(��2conv4_1/kernel
:�2conv4_1/bias
:2norm4_1/gamma
:2norm4_1/beta
*:(��2conv4_2/kernel
:�2conv4_2/bias
:2norm4_2/gamma
:2norm4_2/beta
*:(��2conv5_1/kernel
:�2conv5_1/bias
:2norm5_1/gamma
:2norm5_1/beta
*:(��2conv5_2/kernel
:�2conv5_2/bias
:2norm5_2/gamma
:2norm5_2/beta
*:(��2conv6_1/kernel
:�2conv6_1/bias
:2norm6_1/gamma
:2norm6_1/beta
*:(��2conv6_2/kernel
:�2conv6_2/bias
:2norm6_2/gamma
:2norm6_2/beta
*:(��2conv7_1/kernel
:�2conv7_1/bias
:2norm7_1/gamma
:2norm7_1/beta
*:(��2conv7_2/kernel
:�2conv7_2/bias
:2norm7_2/gamma
:2norm7_2/beta
*:(��2conv8_1/kernel
:�2conv8_1/bias
:2norm8_1/gamma
:2norm8_1/beta
*:(��2conv8_2/kernel
:�2conv8_2/bias
:2norm8_2/gamma
:2norm8_2/beta
*:(��2conv9_1/kernel
:�2conv9_1/bias
:2norm9_1/gamma
:2norm9_1/beta
*:(��2conv9_2/kernel
:�2conv9_2/bias
:2norm9_2/gamma
:2norm9_2/beta
+:)��2conv10_1/kernel
:�2conv10_1/bias
:2norm10_1/gamma
:2norm10_1/beta
+:)��2conv10_2/kernel
:�2conv10_2/bias
:2norm10_2/gamma
:2norm10_2/beta
+:)��2conv11_1/kernel
:�2conv11_1/bias
:2norm11_1/gamma
:2norm11_1/beta
+:)��2conv11_2/kernel
:�2conv11_2/bias
:2norm11_2/gamma
:2norm11_2/beta
,:*��2deconv1_1/kernel
:�2deconv1_1/bias
,:*��2deconv1_2/kernel
:�2deconv1_2/bias
 :2norm_deconv1/gamma
:2norm_deconv1/beta
,:*��2deconv2_1/kernel
:�2deconv2_1/bias
,:*��2deconv2_2/kernel
:�2deconv2_2/bias
 :2norm_deconv2/gamma
:2norm_deconv2/beta
):'�2deconv3/kernel
:2deconv3/bias
�2�
__inference__wrapped_model_7173�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input�����������
/B-
"__inference_signature_wrapper_7276input�
"__inference_signature_wrapper_7276��Z[\]^_`abcdefghijklmnopqrstuvwxyz{|}~��������������������������������������������������������A�>
� 
7�4
2
input)�&
input�����������"G�D
B
activation_131�.
activation_13������������
__inference__wrapped_model_7173��Z[\]^_`abcdefghijklmnopqrstuvwxyz{|}~��������������������������������������������������������8�5
.�+
)�&
input�����������
� "G�D
B
activation_131�.
activation_13�����������