??
?"?"
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.52v1.15.4-39-g3db52be??
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:?????????o*
shape:?????????o
p
Placeholder_1Placeholder*'
_output_shapes
:?????????*
shape:?????????*
dtype0
p
Placeholder_2Placeholder*
dtype0*
shape:?????????o*'
_output_shapes
:?????????o
h
Placeholder_3Placeholder*
shape:?????????*
dtype0*#
_output_shapes
:?????????
h
Placeholder_4Placeholder*
shape:?????????*#
_output_shapes
:?????????*
dtype0
?
5main/pi/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"o      *
dtype0*'
_class
loc:@main/pi/dense/kernel
?
3main/pi/dense/kernel/Initializer/random_uniform/minConst*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *W??
?
3main/pi/dense/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@main/pi/dense/kernel*
valueB
 *W?>*
_output_shapes
: *
dtype0
?
=main/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5main/pi/dense/kernel/Initializer/random_uniform/shape*
seed2*
dtype0*
T0*'
_class
loc:@main/pi/dense/kernel*

seed *
_output_shapes
:	o?
?
3main/pi/dense/kernel/Initializer/random_uniform/subSub3main/pi/dense/kernel/Initializer/random_uniform/max3main/pi/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *'
_class
loc:@main/pi/dense/kernel
?
3main/pi/dense/kernel/Initializer/random_uniform/mulMul=main/pi/dense/kernel/Initializer/random_uniform/RandomUniform3main/pi/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?
?
/main/pi/dense/kernel/Initializer/random_uniformAdd3main/pi/dense/kernel/Initializer/random_uniform/mul3main/pi/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?
?
main/pi/dense/kernel
VariableV2*
dtype0*
	container *'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
shape:	o?*
shared_name 
?
main/pi/dense/kernel/AssignAssignmain/pi/dense/kernel/main/pi/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
T0
?
main/pi/dense/kernel/readIdentitymain/pi/dense/kernel*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
T0
?
$main/pi/dense/bias/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *%
_class
loc:@main/pi/dense/bias
?
main/pi/dense/bias
VariableV2*
shared_name *
	container *
dtype0*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?*
shape:?
?
main/pi/dense/bias/AssignAssignmain/pi/dense/bias$main/pi/dense/bias/Initializer/zeros*
T0*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
use_locking(
?
main/pi/dense/bias/readIdentitymain/pi/dense/bias*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?*
T0
?
main/pi/dense/MatMulMatMulPlaceholdermain/pi/dense/kernel/read*
transpose_a( *
T0*(
_output_shapes
:??????????*
transpose_b( 
?
main/pi/dense/BiasAddBiasAddmain/pi/dense/MatMulmain/pi/dense/bias/read*(
_output_shapes
:??????????*
data_formatNHWC*
T0
d
main/pi/dense/ReluRelumain/pi/dense/BiasAdd*
T0*(
_output_shapes
:??????????
?
7main/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*)
_class
loc:@main/pi/dense_1/kernel*
dtype0
?
5main/pi/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *׳ݽ*)
_class
loc:@main/pi/dense_1/kernel*
dtype0
?
5main/pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳?=*)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
?
?main/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_1/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@main/pi/dense_1/kernel*

seed *
seed2* 
_output_shapes
:
??*
dtype0
?
5main/pi/dense_1/kernel/Initializer/random_uniform/subSub5main/pi/dense_1/kernel/Initializer/random_uniform/max5main/pi/dense_1/kernel/Initializer/random_uniform/min*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
: *
T0
?
5main/pi/dense_1/kernel/Initializer/random_uniform/mulMul?main/pi/dense_1/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_1/kernel/Initializer/random_uniform/sub*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
T0
?
1main/pi/dense_1/kernel/Initializer/random_uniformAdd5main/pi/dense_1/kernel/Initializer/random_uniform/mul5main/pi/dense_1/kernel/Initializer/random_uniform/min*)
_class
loc:@main/pi/dense_1/kernel*
T0* 
_output_shapes
:
??
?
main/pi/dense_1/kernel
VariableV2* 
_output_shapes
:
??*
shape:
??*
	container *)
_class
loc:@main/pi/dense_1/kernel*
shared_name *
dtype0
?
main/pi/dense_1/kernel/AssignAssignmain/pi/dense_1/kernel1main/pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(* 
_output_shapes
:
??*
T0*)
_class
loc:@main/pi/dense_1/kernel
?
main/pi/dense_1/kernel/readIdentitymain/pi/dense_1/kernel* 
_output_shapes
:
??*
T0*)
_class
loc:@main/pi/dense_1/kernel
?
&main/pi/dense_1/bias/Initializer/zerosConst*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
dtype0*
valueB?*    
?
main/pi/dense_1/bias
VariableV2*'
_class
loc:@main/pi/dense_1/bias*
shared_name *
shape:?*
_output_shapes	
:?*
dtype0*
	container 
?
main/pi/dense_1/bias/AssignAssignmain/pi/dense_1/bias&main/pi/dense_1/bias/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
T0
?
main/pi/dense_1/bias/readIdentitymain/pi/dense_1/bias*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:?
?
main/pi/dense_1/MatMulMatMulmain/pi/dense/Relumain/pi/dense_1/kernel/read*
transpose_a( *
T0*
transpose_b( *(
_output_shapes
:??????????
?
main/pi/dense_1/BiasAddBiasAddmain/pi/dense_1/MatMulmain/pi/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:??????????*
T0
h
main/pi/dense_1/ReluRelumain/pi/dense_1/BiasAdd*(
_output_shapes
:??????????*
T0
?
7main/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@main/pi/dense_2/kernel*
dtype0*
_output_shapes
:
?
5main/pi/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *?_?*
dtype0*)
_class
loc:@main/pi/dense_2/kernel
?
5main/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *?_>*
dtype0*)
_class
loc:@main/pi/dense_2/kernel
?
?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_2/kernel/Initializer/random_uniform/shape*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel*
seed2**

seed *
dtype0*
T0
?
5main/pi/dense_2/kernel/Initializer/random_uniform/subSub5main/pi/dense_2/kernel/Initializer/random_uniform/max5main/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *)
_class
loc:@main/pi/dense_2/kernel
?
5main/pi/dense_2/kernel/Initializer/random_uniform/mulMul?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_2/kernel/Initializer/random_uniform/sub*)
_class
loc:@main/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
1main/pi/dense_2/kernel/Initializer/random_uniformAdd5main/pi/dense_2/kernel/Initializer/random_uniform/mul5main/pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	?*
T0*)
_class
loc:@main/pi/dense_2/kernel
?
main/pi/dense_2/kernel
VariableV2*)
_class
loc:@main/pi/dense_2/kernel*
	container *
_output_shapes
:	?*
dtype0*
shared_name *
shape:	?
?
main/pi/dense_2/kernel/AssignAssignmain/pi/dense_2/kernel1main/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
main/pi/dense_2/kernel/readIdentitymain/pi/dense_2/kernel*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel*
T0
?
&main/pi/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
valueB*    *
dtype0
?
main/pi/dense_2/bias
VariableV2*
dtype0*
shape:*'
_class
loc:@main/pi/dense_2/bias*
	container *
_output_shapes
:*
shared_name 
?
main/pi/dense_2/bias/AssignAssignmain/pi/dense_2/bias&main/pi/dense_2/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
T0
?
main/pi/dense_2/bias/readIdentitymain/pi/dense_2/bias*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
T0
?
main/pi/dense_2/MatMulMatMulmain/pi/dense_1/Relumain/pi/dense_2/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:?????????
?
main/pi/dense_2/BiasAddBiasAddmain/pi/dense_2/MatMulmain/pi/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:?????????
g
main/pi/dense_2/TanhTanhmain/pi/dense_2/BiasAdd*'
_output_shapes
:?????????*
T0
R
main/pi/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
i
main/pi/mulMulmain/pi/mul/xmain/pi/dense_2/Tanh*
T0*'
_output_shapes
:?????????
]
main/q/concat/axisConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
main/q/concatConcatV2PlaceholderPlaceholder_1main/q/concat/axis*
N*

Tidx0*
T0*'
_output_shapes
:?????????w
?
4main/q/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"w      *&
_class
loc:@main/q/dense/kernel*
dtype0*
_output_shapes
:
?
2main/q/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *???*&
_class
loc:@main/q/dense/kernel*
dtype0
?
2main/q/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *??>*&
_class
loc:@main/q/dense/kernel*
_output_shapes
: *
dtype0
?
<main/q/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/q/dense/kernel/Initializer/random_uniform/shape*

seed *&
_class
loc:@main/q/dense/kernel*
dtype0*
T0*
_output_shapes
:	w?*
seed2?
?
2main/q/dense/kernel/Initializer/random_uniform/subSub2main/q/dense/kernel/Initializer/random_uniform/max2main/q/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *&
_class
loc:@main/q/dense/kernel
?
2main/q/dense/kernel/Initializer/random_uniform/mulMul<main/q/dense/kernel/Initializer/random_uniform/RandomUniform2main/q/dense/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?
?
.main/q/dense/kernel/Initializer/random_uniformAdd2main/q/dense/kernel/Initializer/random_uniform/mul2main/q/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
T0
?
main/q/dense/kernel
VariableV2*
	container *&
_class
loc:@main/q/dense/kernel*
shape:	w?*
shared_name *
dtype0*
_output_shapes
:	w?
?
main/q/dense/kernel/AssignAssignmain/q/dense/kernel.main/q/dense/kernel/Initializer/random_uniform*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	w?
?
main/q/dense/kernel/readIdentitymain/q/dense/kernel*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?
?
#main/q/dense/bias/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
dtype0
?
main/q/dense/bias
VariableV2*
dtype0*
	container *
shared_name *$
_class
loc:@main/q/dense/bias*
shape:?*
_output_shapes	
:?
?
main/q/dense/bias/AssignAssignmain/q/dense/bias#main/q/dense/bias/Initializer/zeros*
use_locking(*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
main/q/dense/bias/readIdentitymain/q/dense/bias*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes	
:?
?
main/q/dense/MatMulMatMulmain/q/concatmain/q/dense/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:??????????
?
main/q/dense/BiasAddBiasAddmain/q/dense/MatMulmain/q/dense/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:??????????
b
main/q/dense/ReluRelumain/q/dense/BiasAdd*(
_output_shapes
:??????????*
T0
?
6main/q/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*(
_class
loc:@main/q/dense_1/kernel*
valueB"      
?
4main/q/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *׳ݽ*(
_class
loc:@main/q/dense_1/kernel*
dtype0
?
4main/q/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*(
_class
loc:@main/q/dense_1/kernel*
valueB
 *׳?=
?
>main/q/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform6main/q/dense_1/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
??*
T0*
seed2P*
dtype0*

seed *(
_class
loc:@main/q/dense_1/kernel
?
4main/q/dense_1/kernel/Initializer/random_uniform/subSub4main/q/dense_1/kernel/Initializer/random_uniform/max4main/q/dense_1/kernel/Initializer/random_uniform/min*(
_class
loc:@main/q/dense_1/kernel*
T0*
_output_shapes
: 
?
4main/q/dense_1/kernel/Initializer/random_uniform/mulMul>main/q/dense_1/kernel/Initializer/random_uniform/RandomUniform4main/q/dense_1/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel
?
0main/q/dense_1/kernel/Initializer/random_uniformAdd4main/q/dense_1/kernel/Initializer/random_uniform/mul4main/q/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
T0
?
main/q/dense_1/kernel
VariableV2*
	container *(
_class
loc:@main/q/dense_1/kernel*
shared_name *
shape:
??*
dtype0* 
_output_shapes
:
??
?
main/q/dense_1/kernel/AssignAssignmain/q/dense_1/kernel0main/q/dense_1/kernel/Initializer/random_uniform*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
T0*
use_locking(
?
main/q/dense_1/kernel/readIdentitymain/q/dense_1/kernel*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
??*
T0
?
%main/q/dense_1/bias/Initializer/zerosConst*
dtype0*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
valueB?*    
?
main/q/dense_1/bias
VariableV2*&
_class
loc:@main/q/dense_1/bias*
	container *
dtype0*
shape:?*
shared_name *
_output_shapes	
:?
?
main/q/dense_1/bias/AssignAssignmain/q/dense_1/bias%main/q/dense_1/bias/Initializer/zeros*
use_locking(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
main/q/dense_1/bias/readIdentitymain/q/dense_1/bias*
_output_shapes	
:?*
T0*&
_class
loc:@main/q/dense_1/bias
?
main/q/dense_1/MatMulMatMulmain/q/dense/Relumain/q/dense_1/kernel/read*
transpose_b( *
transpose_a( *(
_output_shapes
:??????????*
T0
?
main/q/dense_1/BiasAddBiasAddmain/q/dense_1/MatMulmain/q/dense_1/bias/read*(
_output_shapes
:??????????*
T0*
data_formatNHWC
f
main/q/dense_1/ReluRelumain/q/dense_1/BiasAdd*(
_output_shapes
:??????????*
T0
?
6main/q/dense_2/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@main/q/dense_2/kernel*
dtype0*
valueB"      *
_output_shapes
:
?
4main/q/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *(
_class
loc:@main/q/dense_2/kernel*
dtype0*
valueB
 *Iv?
?
4main/q/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*
dtype0*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
: 
?
>main/q/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform6main/q/dense_2/kernel/Initializer/random_uniform/shape*
seed2a*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
T0*

seed *
dtype0
?
4main/q/dense_2/kernel/Initializer/random_uniform/subSub4main/q/dense_2/kernel/Initializer/random_uniform/max4main/q/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *(
_class
loc:@main/q/dense_2/kernel
?
4main/q/dense_2/kernel/Initializer/random_uniform/mulMul>main/q/dense_2/kernel/Initializer/random_uniform/RandomUniform4main/q/dense_2/kernel/Initializer/random_uniform/sub*(
_class
loc:@main/q/dense_2/kernel*
T0*
_output_shapes
:	?
?
0main/q/dense_2/kernel/Initializer/random_uniformAdd4main/q/dense_2/kernel/Initializer/random_uniform/mul4main/q/dense_2/kernel/Initializer/random_uniform/min*(
_class
loc:@main/q/dense_2/kernel*
T0*
_output_shapes
:	?
?
main/q/dense_2/kernel
VariableV2*
	container *
_output_shapes
:	?*
shape:	?*
dtype0*(
_class
loc:@main/q/dense_2/kernel*
shared_name 
?
main/q/dense_2/kernel/AssignAssignmain/q/dense_2/kernel0main/q/dense_2/kernel/Initializer/random_uniform*
T0*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	?
?
main/q/dense_2/kernel/readIdentitymain/q/dense_2/kernel*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
T0
?
%main/q/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias*
valueB*    *
dtype0
?
main/q/dense_2/bias
VariableV2*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias*
dtype0*
shared_name *
shape:*
	container 
?
main/q/dense_2/bias/AssignAssignmain/q/dense_2/bias%main/q/dense_2/bias/Initializer/zeros*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0
?
main/q/dense_2/bias/readIdentitymain/q/dense_2/bias*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
T0
?
main/q/dense_2/MatMulMatMulmain/q/dense_1/Relumain/q/dense_2/kernel/read*
T0*
transpose_b( *'
_output_shapes
:?????????*
transpose_a( 
?
main/q/dense_2/BiasAddBiasAddmain/q/dense_2/MatMulmain/q/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????
v
main/q/SqueezeSqueezemain/q/dense_2/BiasAdd*
T0*
squeeze_dims
*#
_output_shapes
:?????????
_
main/q_1/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
main/q_1/concatConcatV2Placeholdermain/pi/mulmain/q_1/concat/axis*
N*
T0*'
_output_shapes
:?????????w*

Tidx0
?
main/q_1/dense/MatMulMatMulmain/q_1/concatmain/q/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:??????????*
transpose_a( 
?
main/q_1/dense/BiasAddBiasAddmain/q_1/dense/MatMulmain/q/dense/bias/read*
T0*(
_output_shapes
:??????????*
data_formatNHWC
f
main/q_1/dense/ReluRelumain/q_1/dense/BiasAdd*
T0*(
_output_shapes
:??????????
?
main/q_1/dense_1/MatMulMatMulmain/q_1/dense/Relumain/q/dense_1/kernel/read*
transpose_a( *
transpose_b( *(
_output_shapes
:??????????*
T0
?
main/q_1/dense_1/BiasAddBiasAddmain/q_1/dense_1/MatMulmain/q/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:??????????
j
main/q_1/dense_1/ReluRelumain/q_1/dense_1/BiasAdd*
T0*(
_output_shapes
:??????????
?
main/q_1/dense_2/MatMulMatMulmain/q_1/dense_1/Relumain/q/dense_2/kernel/read*
T0*
transpose_b( *'
_output_shapes
:?????????*
transpose_a( 
?
main/q_1/dense_2/BiasAddBiasAddmain/q_1/dense_2/MatMulmain/q/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:?????????
z
main/q_1/SqueezeSqueezemain/q_1/dense_2/BiasAdd*
T0*
squeeze_dims
*#
_output_shapes
:?????????
?
7target/pi/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"o      *)
_class
loc:@target/pi/dense/kernel*
dtype0
?
5target/pi/dense/kernel/Initializer/random_uniform/minConst*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
: *
valueB
 *W??*
dtype0
?
5target/pi/dense/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@target/pi/dense/kernel*
valueB
 *W?>*
dtype0*
_output_shapes
: 
?
?target/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7target/pi/dense/kernel/Initializer/random_uniform/shape*
T0*
seed2}*
dtype0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?*

seed 
?
5target/pi/dense/kernel/Initializer/random_uniform/subSub5target/pi/dense/kernel/Initializer/random_uniform/max5target/pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@target/pi/dense/kernel
?
5target/pi/dense/kernel/Initializer/random_uniform/mulMul?target/pi/dense/kernel/Initializer/random_uniform/RandomUniform5target/pi/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?
?
1target/pi/dense/kernel/Initializer/random_uniformAdd5target/pi/dense/kernel/Initializer/random_uniform/mul5target/pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	o?*)
_class
loc:@target/pi/dense/kernel*
T0
?
target/pi/dense/kernel
VariableV2*
dtype0*
shared_name *
_output_shapes
:	o?*
	container *)
_class
loc:@target/pi/dense/kernel*
shape:	o?
?
target/pi/dense/kernel/AssignAssigntarget/pi/dense/kernel1target/pi/dense/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?
?
target/pi/dense/kernel/readIdentitytarget/pi/dense/kernel*
_output_shapes
:	o?*
T0*)
_class
loc:@target/pi/dense/kernel
?
&target/pi/dense/bias/Initializer/zerosConst*'
_class
loc:@target/pi/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
target/pi/dense/bias
VariableV2*
	container *
dtype0*
shape:?*
_output_shapes	
:?*'
_class
loc:@target/pi/dense/bias*
shared_name 
?
target/pi/dense/bias/AssignAssigntarget/pi/dense/bias&target/pi/dense/bias/Initializer/zeros*'
_class
loc:@target/pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0
?
target/pi/dense/bias/readIdentitytarget/pi/dense/bias*
_output_shapes	
:?*'
_class
loc:@target/pi/dense/bias*
T0
?
target/pi/dense/MatMulMatMulPlaceholder_2target/pi/dense/kernel/read*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( *
T0
?
target/pi/dense/BiasAddBiasAddtarget/pi/dense/MatMultarget/pi/dense/bias/read*(
_output_shapes
:??????????*
T0*
data_formatNHWC
h
target/pi/dense/ReluRelutarget/pi/dense/BiasAdd*
T0*(
_output_shapes
:??????????
?
9target/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
valueB"      
?
7target/pi/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *׳ݽ*+
_class!
loc:@target/pi/dense_1/kernel*
_output_shapes
: *
dtype0
?
7target/pi/dense_1/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *׳?=
?
Atarget/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_1/kernel/Initializer/random_uniform/shape*

seed *
seed2?* 
_output_shapes
:
??*
T0*
dtype0*+
_class!
loc:@target/pi/dense_1/kernel
?
7target/pi/dense_1/kernel/Initializer/random_uniform/subSub7target/pi/dense_1/kernel/Initializer/random_uniform/max7target/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *+
_class!
loc:@target/pi/dense_1/kernel
?
7target/pi/dense_1/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_1/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_1/kernel/Initializer/random_uniform/sub*+
_class!
loc:@target/pi/dense_1/kernel*
T0* 
_output_shapes
:
??
?
3target/pi/dense_1/kernel/Initializer/random_uniformAdd7target/pi/dense_1/kernel/Initializer/random_uniform/mul7target/pi/dense_1/kernel/Initializer/random_uniform/min*+
_class!
loc:@target/pi/dense_1/kernel*
T0* 
_output_shapes
:
??
?
target/pi/dense_1/kernel
VariableV2*+
_class!
loc:@target/pi/dense_1/kernel*
shape:
??* 
_output_shapes
:
??*
shared_name *
	container *
dtype0
?
target/pi/dense_1/kernel/AssignAssigntarget/pi/dense_1/kernel3target/pi/dense_1/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
??*+
_class!
loc:@target/pi/dense_1/kernel
?
target/pi/dense_1/kernel/readIdentitytarget/pi/dense_1/kernel* 
_output_shapes
:
??*
T0*+
_class!
loc:@target/pi/dense_1/kernel
?
(target/pi/dense_1/bias/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*)
_class
loc:@target/pi/dense_1/bias*
dtype0
?
target/pi/dense_1/bias
VariableV2*
	container *
_output_shapes	
:?*)
_class
loc:@target/pi/dense_1/bias*
shape:?*
shared_name *
dtype0
?
target/pi/dense_1/bias/AssignAssigntarget/pi/dense_1/bias(target/pi/dense_1/bias/Initializer/zeros*
_output_shapes	
:?*
T0*)
_class
loc:@target/pi/dense_1/bias*
use_locking(*
validate_shape(
?
target/pi/dense_1/bias/readIdentitytarget/pi/dense_1/bias*
_output_shapes	
:?*
T0*)
_class
loc:@target/pi/dense_1/bias
?
target/pi/dense_1/MatMulMatMultarget/pi/dense/Relutarget/pi/dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:??????????
?
target/pi/dense_1/BiasAddBiasAddtarget/pi/dense_1/MatMultarget/pi/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:??????????
l
target/pi/dense_1/ReluRelutarget/pi/dense_1/BiasAdd*(
_output_shapes
:??????????*
T0
?
9target/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:
?
7target/pi/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *?_?*+
_class!
loc:@target/pi/dense_2/kernel
?
7target/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *?_>*+
_class!
loc:@target/pi/dense_2/kernel*
dtype0
?
Atarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_2/kernel/Initializer/random_uniform/shape*
T0*
dtype0*

seed *+
_class!
loc:@target/pi/dense_2/kernel*
seed2?*
_output_shapes
:	?
?
7target/pi/dense_2/kernel/Initializer/random_uniform/subSub7target/pi/dense_2/kernel/Initializer/random_uniform/max7target/pi/dense_2/kernel/Initializer/random_uniform/min*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
: *
T0
?
7target/pi/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_2/kernel/Initializer/random_uniform/sub*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
3target/pi/dense_2/kernel/Initializer/random_uniformAdd7target/pi/dense_2/kernel/Initializer/random_uniform/mul7target/pi/dense_2/kernel/Initializer/random_uniform/min*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
target/pi/dense_2/kernel
VariableV2*
dtype0*
shared_name *
_output_shapes
:	?*+
_class!
loc:@target/pi/dense_2/kernel*
	container *
shape:	?
?
target/pi/dense_2/kernel/AssignAssigntarget/pi/dense_2/kernel3target/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	?*
T0
?
target/pi/dense_2/kernel/readIdentitytarget/pi/dense_2/kernel*
_output_shapes
:	?*+
_class!
loc:@target/pi/dense_2/kernel*
T0
?
(target/pi/dense_2/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:*
dtype0
?
target/pi/dense_2/bias
VariableV2*
	container *
_output_shapes
:*
shape:*)
_class
loc:@target/pi/dense_2/bias*
dtype0*
shared_name 
?
target/pi/dense_2/bias/AssignAssigntarget/pi/dense_2/bias(target/pi/dense_2/bias/Initializer/zeros*
T0*
use_locking(*)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:*
validate_shape(
?
target/pi/dense_2/bias/readIdentitytarget/pi/dense_2/bias*
_output_shapes
:*
T0*)
_class
loc:@target/pi/dense_2/bias
?
target/pi/dense_2/MatMulMatMultarget/pi/dense_1/Relutarget/pi/dense_2/kernel/read*
transpose_b( *
transpose_a( *'
_output_shapes
:?????????*
T0
?
target/pi/dense_2/BiasAddBiasAddtarget/pi/dense_2/MatMultarget/pi/dense_2/bias/read*'
_output_shapes
:?????????*
data_formatNHWC*
T0
k
target/pi/dense_2/TanhTanhtarget/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:?????????
T
target/pi/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
o
target/pi/mulMultarget/pi/mul/xtarget/pi/dense_2/Tanh*
T0*'
_output_shapes
:?????????
_
target/q/concat/axisConst*
valueB :
?????????*
_output_shapes
: *
dtype0
?
target/q/concatConcatV2Placeholder_2Placeholder_1target/q/concat/axis*'
_output_shapes
:?????????w*
N*
T0*

Tidx0
?
6target/q/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"w      *
_output_shapes
:*(
_class
loc:@target/q/dense/kernel
?
4target/q/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *(
_class
loc:@target/q/dense/kernel*
dtype0*
valueB
 *???
?
4target/q/dense/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/q/dense/kernel*
valueB
 *??>*
_output_shapes
: *
dtype0
?
>target/q/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/q/dense/kernel/Initializer/random_uniform/shape*
seed2?*
T0*
_output_shapes
:	w?*
dtype0*(
_class
loc:@target/q/dense/kernel*

seed 
?
4target/q/dense/kernel/Initializer/random_uniform/subSub4target/q/dense/kernel/Initializer/random_uniform/max4target/q/dense/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/q/dense/kernel*
_output_shapes
: 
?
4target/q/dense/kernel/Initializer/random_uniform/mulMul>target/q/dense/kernel/Initializer/random_uniform/RandomUniform4target/q/dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	w?*(
_class
loc:@target/q/dense/kernel
?
0target/q/dense/kernel/Initializer/random_uniformAdd4target/q/dense/kernel/Initializer/random_uniform/mul4target/q/dense/kernel/Initializer/random_uniform/min*(
_class
loc:@target/q/dense/kernel*
T0*
_output_shapes
:	w?
?
target/q/dense/kernel
VariableV2*
dtype0*
	container *
_output_shapes
:	w?*(
_class
loc:@target/q/dense/kernel*
shared_name *
shape:	w?
?
target/q/dense/kernel/AssignAssigntarget/q/dense/kernel0target/q/dense/kernel/Initializer/random_uniform*(
_class
loc:@target/q/dense/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	w?
?
target/q/dense/kernel/readIdentitytarget/q/dense/kernel*
_output_shapes
:	w?*(
_class
loc:@target/q/dense/kernel*
T0
?
%target/q/dense/bias/Initializer/zerosConst*&
_class
loc:@target/q/dense/bias*
valueB?*    *
_output_shapes	
:?*
dtype0
?
target/q/dense/bias
VariableV2*
shape:?*
shared_name *
_output_shapes	
:?*
	container *&
_class
loc:@target/q/dense/bias*
dtype0
?
target/q/dense/bias/AssignAssigntarget/q/dense/bias%target/q/dense/bias/Initializer/zeros*&
_class
loc:@target/q/dense/bias*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0
?
target/q/dense/bias/readIdentitytarget/q/dense/bias*
_output_shapes	
:?*
T0*&
_class
loc:@target/q/dense/bias
?
target/q/dense/MatMulMatMultarget/q/concattarget/q/dense/kernel/read*
transpose_b( *(
_output_shapes
:??????????*
T0*
transpose_a( 
?
target/q/dense/BiasAddBiasAddtarget/q/dense/MatMultarget/q/dense/bias/read*(
_output_shapes
:??????????*
data_formatNHWC*
T0
f
target/q/dense/ReluRelutarget/q/dense/BiasAdd*
T0*(
_output_shapes
:??????????
?
8target/q/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0**
_class 
loc:@target/q/dense_1/kernel*
_output_shapes
:
?
6target/q/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *׳ݽ*
_output_shapes
: **
_class 
loc:@target/q/dense_1/kernel
?
6target/q/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳?=**
_class 
loc:@target/q/dense_1/kernel*
_output_shapes
: *
dtype0
?
@target/q/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform8target/q/dense_1/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
??**
_class 
loc:@target/q/dense_1/kernel*

seed *
seed2?*
dtype0*
T0
?
6target/q/dense_1/kernel/Initializer/random_uniform/subSub6target/q/dense_1/kernel/Initializer/random_uniform/max6target/q/dense_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@target/q/dense_1/kernel*
_output_shapes
: 
?
6target/q/dense_1/kernel/Initializer/random_uniform/mulMul@target/q/dense_1/kernel/Initializer/random_uniform/RandomUniform6target/q/dense_1/kernel/Initializer/random_uniform/sub**
_class 
loc:@target/q/dense_1/kernel*
T0* 
_output_shapes
:
??
?
2target/q/dense_1/kernel/Initializer/random_uniformAdd6target/q/dense_1/kernel/Initializer/random_uniform/mul6target/q/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
??*
T0**
_class 
loc:@target/q/dense_1/kernel
?
target/q/dense_1/kernel
VariableV2**
_class 
loc:@target/q/dense_1/kernel*
shared_name * 
_output_shapes
:
??*
dtype0*
	container *
shape:
??
?
target/q/dense_1/kernel/AssignAssigntarget/q/dense_1/kernel2target/q/dense_1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
??*
T0**
_class 
loc:@target/q/dense_1/kernel*
use_locking(
?
target/q/dense_1/kernel/readIdentitytarget/q/dense_1/kernel*
T0**
_class 
loc:@target/q/dense_1/kernel* 
_output_shapes
:
??
?
'target/q/dense_1/bias/Initializer/zerosConst*
dtype0*(
_class
loc:@target/q/dense_1/bias*
valueB?*    *
_output_shapes	
:?
?
target/q/dense_1/bias
VariableV2*
shape:?*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias*
	container *
shared_name *
dtype0
?
target/q/dense_1/bias/AssignAssigntarget/q/dense_1/bias'target/q/dense_1/bias/Initializer/zeros*
use_locking(*(
_class
loc:@target/q/dense_1/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
target/q/dense_1/bias/readIdentitytarget/q/dense_1/bias*
T0*(
_class
loc:@target/q/dense_1/bias*
_output_shapes	
:?
?
target/q/dense_1/MatMulMatMultarget/q/dense/Relutarget/q/dense_1/kernel/read*(
_output_shapes
:??????????*
T0*
transpose_a( *
transpose_b( 
?
target/q/dense_1/BiasAddBiasAddtarget/q/dense_1/MatMultarget/q/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:??????????*
T0
j
target/q/dense_1/ReluRelutarget/q/dense_1/BiasAdd*
T0*(
_output_shapes
:??????????
?
8target/q/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"      **
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:
?
6target/q/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
: *
valueB
 *Iv?
?
6target/q/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *Iv>*
_output_shapes
: **
_class 
loc:@target/q/dense_2/kernel
?
@target/q/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform8target/q/dense_2/kernel/Initializer/random_uniform/shape*
_output_shapes
:	?*
T0*
dtype0*
seed2?*

seed **
_class 
loc:@target/q/dense_2/kernel
?
6target/q/dense_2/kernel/Initializer/random_uniform/subSub6target/q/dense_2/kernel/Initializer/random_uniform/max6target/q/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: **
_class 
loc:@target/q/dense_2/kernel
?
6target/q/dense_2/kernel/Initializer/random_uniform/mulMul@target/q/dense_2/kernel/Initializer/random_uniform/RandomUniform6target/q/dense_2/kernel/Initializer/random_uniform/sub**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:	?*
T0
?
2target/q/dense_2/kernel/Initializer/random_uniformAdd6target/q/dense_2/kernel/Initializer/random_uniform/mul6target/q/dense_2/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:	?
?
target/q/dense_2/kernel
VariableV2*
_output_shapes
:	?**
_class 
loc:@target/q/dense_2/kernel*
	container *
shape:	?*
dtype0*
shared_name 
?
target/q/dense_2/kernel/AssignAssigntarget/q/dense_2/kernel2target/q/dense_2/kernel/Initializer/random_uniform*
T0**
_class 
loc:@target/q/dense_2/kernel*
use_locking(*
_output_shapes
:	?*
validate_shape(
?
target/q/dense_2/kernel/readIdentitytarget/q/dense_2/kernel*
T0**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:	?
?
'target/q/dense_2/bias/Initializer/zerosConst*(
_class
loc:@target/q/dense_2/bias*
_output_shapes
:*
dtype0*
valueB*    
?
target/q/dense_2/bias
VariableV2*(
_class
loc:@target/q/dense_2/bias*
dtype0*
	container *
shared_name *
_output_shapes
:*
shape:
?
target/q/dense_2/bias/AssignAssigntarget/q/dense_2/bias'target/q/dense_2/bias/Initializer/zeros*(
_class
loc:@target/q/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
?
target/q/dense_2/bias/readIdentitytarget/q/dense_2/bias*(
_class
loc:@target/q/dense_2/bias*
_output_shapes
:*
T0
?
target/q/dense_2/MatMulMatMultarget/q/dense_1/Relutarget/q/dense_2/kernel/read*'
_output_shapes
:?????????*
T0*
transpose_a( *
transpose_b( 
?
target/q/dense_2/BiasAddBiasAddtarget/q/dense_2/MatMultarget/q/dense_2/bias/read*'
_output_shapes
:?????????*
T0*
data_formatNHWC
z
target/q/SqueezeSqueezetarget/q/dense_2/BiasAdd*
squeeze_dims
*#
_output_shapes
:?????????*
T0
a
target/q_1/concat/axisConst*
_output_shapes
: *
valueB :
?????????*
dtype0
?
target/q_1/concatConcatV2Placeholder_2target/pi/multarget/q_1/concat/axis*

Tidx0*
N*'
_output_shapes
:?????????w*
T0
?
target/q_1/dense/MatMulMatMultarget/q_1/concattarget/q/dense/kernel/read*
transpose_a( *
transpose_b( *(
_output_shapes
:??????????*
T0
?
target/q_1/dense/BiasAddBiasAddtarget/q_1/dense/MatMultarget/q/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:??????????
j
target/q_1/dense/ReluRelutarget/q_1/dense/BiasAdd*
T0*(
_output_shapes
:??????????
?
target/q_1/dense_1/MatMulMatMultarget/q_1/dense/Relutarget/q/dense_1/kernel/read*
transpose_a( *
transpose_b( *(
_output_shapes
:??????????*
T0
?
target/q_1/dense_1/BiasAddBiasAddtarget/q_1/dense_1/MatMultarget/q/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:??????????
n
target/q_1/dense_1/ReluRelutarget/q_1/dense_1/BiasAdd*(
_output_shapes
:??????????*
T0
?
target/q_1/dense_2/MatMulMatMultarget/q_1/dense_1/Relutarget/q/dense_2/kernel/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:?????????
?
target/q_1/dense_2/BiasAddBiasAddtarget/q_1/dense_2/MatMultarget/q/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????
~
target/q_1/SqueezeSqueezetarget/q_1/dense_2/BiasAdd*
T0*#
_output_shapes
:?????????*
squeeze_dims

J
sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
N
subSubsub/xPlaceholder_4*
T0*#
_output_shapes
:?????????
J
mul/xConst*
valueB
 *ףp?*
dtype0*
_output_shapes
: 
D
mulMulmul/xsub*#
_output_shapes
:?????????*
T0
S
mul_1Mulmultarget/q_1/Squeeze*#
_output_shapes
:?????????*
T0
P
addAddV2Placeholder_3mul_1*
T0*#
_output_shapes
:?????????
O
StopGradientStopGradientadd*
T0*#
_output_shapes
:?????????
O
ConstConst*
valueB: *
_output_shapes
:*
dtype0
c
MeanMeanmain/q_1/SqueezeConst*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
1
NegNegMean*
T0*
_output_shapes
: 
X
sub_1Submain/q/SqueezeStopGradient*#
_output_shapes
:?????????*
T0
J
pow/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
F
powPowsub_1pow/y*
T0*#
_output_shapes
:?????????
Q
Const_1Const*
valueB: *
_output_shapes
:*
dtype0
Z
Mean_1MeanpowConst_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
?
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
i
gradients/Mean_grad/ShapeShapemain/q_1/Squeeze*
_output_shapes
:*
out_type0*
T0
?
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*#
_output_shapes
:?????????*
T0
k
gradients/Mean_grad/Shape_1Shapemain/q_1/Squeeze*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
c
gradients/Mean_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
?
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
?
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
?
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
?
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *
Truncate( *

DstT0*

SrcT0
?
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:?????????*
T0
}
%gradients/main/q_1/Squeeze_grad/ShapeShapemain/q_1/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
?
'gradients/main/q_1/Squeeze_grad/ReshapeReshapegradients/Mean_grad/truediv%gradients/main/q_1/Squeeze_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
?
3gradients/main/q_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/main/q_1/Squeeze_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
?
8gradients/main/q_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp(^gradients/main/q_1/Squeeze_grad/Reshape4^gradients/main/q_1/dense_2/BiasAdd_grad/BiasAddGrad
?
@gradients/main/q_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/main/q_1/Squeeze_grad/Reshape9^gradients/main/q_1/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:?????????*:
_class0
.,loc:@gradients/main/q_1/Squeeze_grad/Reshape*
T0
?
Bgradients/main/q_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/main/q_1/dense_2/BiasAdd_grad/BiasAddGrad9^gradients/main/q_1/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*F
_class<
:8loc:@gradients/main/q_1/dense_2/BiasAdd_grad/BiasAddGrad
?
-gradients/main/q_1/dense_2/MatMul_grad/MatMulMatMul@gradients/main/q_1/dense_2/BiasAdd_grad/tuple/control_dependencymain/q/dense_2/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b(
?
/gradients/main/q_1/dense_2/MatMul_grad/MatMul_1MatMulmain/q_1/dense_1/Relu@gradients/main/q_1/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	?
?
7gradients/main/q_1/dense_2/MatMul_grad/tuple/group_depsNoOp.^gradients/main/q_1/dense_2/MatMul_grad/MatMul0^gradients/main/q_1/dense_2/MatMul_grad/MatMul_1
?
?gradients/main/q_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity-gradients/main/q_1/dense_2/MatMul_grad/MatMul8^gradients/main/q_1/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:??????????*@
_class6
42loc:@gradients/main/q_1/dense_2/MatMul_grad/MatMul*
T0
?
Agradients/main/q_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity/gradients/main/q_1/dense_2/MatMul_grad/MatMul_18^gradients/main/q_1/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	?*B
_class8
64loc:@gradients/main/q_1/dense_2/MatMul_grad/MatMul_1
?
-gradients/main/q_1/dense_1/Relu_grad/ReluGradReluGrad?gradients/main/q_1/dense_2/MatMul_grad/tuple/control_dependencymain/q_1/dense_1/Relu*
T0*(
_output_shapes
:??????????
?
3gradients/main/q_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/main/q_1/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:?*
T0
?
8gradients/main/q_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^gradients/main/q_1/dense_1/BiasAdd_grad/BiasAddGrad.^gradients/main/q_1/dense_1/Relu_grad/ReluGrad
?
@gradients/main/q_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/main/q_1/dense_1/Relu_grad/ReluGrad9^gradients/main/q_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:??????????*@
_class6
42loc:@gradients/main/q_1/dense_1/Relu_grad/ReluGrad
?
Bgradients/main/q_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/main/q_1/dense_1/BiasAdd_grad/BiasAddGrad9^gradients/main/q_1/dense_1/BiasAdd_grad/tuple/group_deps*F
_class<
:8loc:@gradients/main/q_1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?*
T0
?
-gradients/main/q_1/dense_1/MatMul_grad/MatMulMatMul@gradients/main/q_1/dense_1/BiasAdd_grad/tuple/control_dependencymain/q/dense_1/kernel/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:??????????
?
/gradients/main/q_1/dense_1/MatMul_grad/MatMul_1MatMulmain/q_1/dense/Relu@gradients/main/q_1/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
??*
transpose_b( *
transpose_a(*
T0
?
7gradients/main/q_1/dense_1/MatMul_grad/tuple/group_depsNoOp.^gradients/main/q_1/dense_1/MatMul_grad/MatMul0^gradients/main/q_1/dense_1/MatMul_grad/MatMul_1
?
?gradients/main/q_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity-gradients/main/q_1/dense_1/MatMul_grad/MatMul8^gradients/main/q_1/dense_1/MatMul_grad/tuple/group_deps*@
_class6
42loc:@gradients/main/q_1/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
Agradients/main/q_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity/gradients/main/q_1/dense_1/MatMul_grad/MatMul_18^gradients/main/q_1/dense_1/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
??*B
_class8
64loc:@gradients/main/q_1/dense_1/MatMul_grad/MatMul_1
?
+gradients/main/q_1/dense/Relu_grad/ReluGradReluGrad?gradients/main/q_1/dense_1/MatMul_grad/tuple/control_dependencymain/q_1/dense/Relu*
T0*(
_output_shapes
:??????????
?
1gradients/main/q_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients/main/q_1/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:?*
T0
?
6gradients/main/q_1/dense/BiasAdd_grad/tuple/group_depsNoOp2^gradients/main/q_1/dense/BiasAdd_grad/BiasAddGrad,^gradients/main/q_1/dense/Relu_grad/ReluGrad
?
>gradients/main/q_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity+gradients/main/q_1/dense/Relu_grad/ReluGrad7^gradients/main/q_1/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/q_1/dense/Relu_grad/ReluGrad*(
_output_shapes
:??????????
?
@gradients/main/q_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/main/q_1/dense/BiasAdd_grad/BiasAddGrad7^gradients/main/q_1/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/main/q_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
+gradients/main/q_1/dense/MatMul_grad/MatMulMatMul>gradients/main/q_1/dense/BiasAdd_grad/tuple/control_dependencymain/q/dense/kernel/read*
transpose_b(*'
_output_shapes
:?????????w*
transpose_a( *
T0
?
-gradients/main/q_1/dense/MatMul_grad/MatMul_1MatMulmain/q_1/concat>gradients/main/q_1/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	w?*
T0*
transpose_b( 
?
5gradients/main/q_1/dense/MatMul_grad/tuple/group_depsNoOp,^gradients/main/q_1/dense/MatMul_grad/MatMul.^gradients/main/q_1/dense/MatMul_grad/MatMul_1
?
=gradients/main/q_1/dense/MatMul_grad/tuple/control_dependencyIdentity+gradients/main/q_1/dense/MatMul_grad/MatMul6^gradients/main/q_1/dense/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/q_1/dense/MatMul_grad/MatMul*'
_output_shapes
:?????????w
?
?gradients/main/q_1/dense/MatMul_grad/tuple/control_dependency_1Identity-gradients/main/q_1/dense/MatMul_grad/MatMul_16^gradients/main/q_1/dense/MatMul_grad/tuple/group_deps*@
_class6
42loc:@gradients/main/q_1/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	w?
e
#gradients/main/q_1/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
?
"gradients/main/q_1/concat_grad/modFloorModmain/q_1/concat/axis#gradients/main/q_1/concat_grad/Rank*
_output_shapes
: *
T0
o
$gradients/main/q_1/concat_grad/ShapeShapePlaceholder*
out_type0*
T0*
_output_shapes
:
?
%gradients/main/q_1/concat_grad/ShapeNShapeNPlaceholdermain/pi/mul*
out_type0* 
_output_shapes
::*
T0*
N
?
+gradients/main/q_1/concat_grad/ConcatOffsetConcatOffset"gradients/main/q_1/concat_grad/mod%gradients/main/q_1/concat_grad/ShapeN'gradients/main/q_1/concat_grad/ShapeN:1*
N* 
_output_shapes
::
?
$gradients/main/q_1/concat_grad/SliceSlice=gradients/main/q_1/dense/MatMul_grad/tuple/control_dependency+gradients/main/q_1/concat_grad/ConcatOffset%gradients/main/q_1/concat_grad/ShapeN*
T0*'
_output_shapes
:?????????o*
Index0
?
&gradients/main/q_1/concat_grad/Slice_1Slice=gradients/main/q_1/dense/MatMul_grad/tuple/control_dependency-gradients/main/q_1/concat_grad/ConcatOffset:1'gradients/main/q_1/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:?????????
?
/gradients/main/q_1/concat_grad/tuple/group_depsNoOp%^gradients/main/q_1/concat_grad/Slice'^gradients/main/q_1/concat_grad/Slice_1
?
7gradients/main/q_1/concat_grad/tuple/control_dependencyIdentity$gradients/main/q_1/concat_grad/Slice0^gradients/main/q_1/concat_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/q_1/concat_grad/Slice*'
_output_shapes
:?????????o
?
9gradients/main/q_1/concat_grad/tuple/control_dependency_1Identity&gradients/main/q_1/concat_grad/Slice_10^gradients/main/q_1/concat_grad/tuple/group_deps*9
_class/
-+loc:@gradients/main/q_1/concat_grad/Slice_1*'
_output_shapes
:?????????*
T0
k
 gradients/main/pi/mul_grad/ShapeShapemain/pi/mul/x*
out_type0*
_output_shapes
: *
T0
v
"gradients/main/pi/mul_grad/Shape_1Shapemain/pi/dense_2/Tanh*
T0*
out_type0*
_output_shapes
:
?
0gradients/main/pi/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/main/pi/mul_grad/Shape"gradients/main/pi/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/main/pi/mul_grad/MulMul9gradients/main/q_1/concat_grad/tuple/control_dependency_1main/pi/dense_2/Tanh*'
_output_shapes
:?????????*
T0
?
gradients/main/pi/mul_grad/SumSumgradients/main/pi/mul_grad/Mul0gradients/main/pi/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
"gradients/main/pi/mul_grad/ReshapeReshapegradients/main/pi/mul_grad/Sum gradients/main/pi/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
?
 gradients/main/pi/mul_grad/Mul_1Mulmain/pi/mul/x9gradients/main/q_1/concat_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
 gradients/main/pi/mul_grad/Sum_1Sum gradients/main/pi/mul_grad/Mul_12gradients/main/pi/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
$gradients/main/pi/mul_grad/Reshape_1Reshape gradients/main/pi/mul_grad/Sum_1"gradients/main/pi/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????*
T0

+gradients/main/pi/mul_grad/tuple/group_depsNoOp#^gradients/main/pi/mul_grad/Reshape%^gradients/main/pi/mul_grad/Reshape_1
?
3gradients/main/pi/mul_grad/tuple/control_dependencyIdentity"gradients/main/pi/mul_grad/Reshape,^gradients/main/pi/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/main/pi/mul_grad/Reshape*
_output_shapes
: 
?
5gradients/main/pi/mul_grad/tuple/control_dependency_1Identity$gradients/main/pi/mul_grad/Reshape_1,^gradients/main/pi/mul_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*7
_class-
+)loc:@gradients/main/pi/mul_grad/Reshape_1
?
,gradients/main/pi/dense_2/Tanh_grad/TanhGradTanhGradmain/pi/dense_2/Tanh5gradients/main/pi/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
2gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/main/pi/dense_2/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes
:
?
7gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad-^gradients/main/pi/dense_2/Tanh_grad/TanhGrad
?
?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_2/Tanh_grad/TanhGrad8^gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:?????????*?
_class5
31loc:@gradients/main/pi/dense_2/Tanh_grad/TanhGrad*
T0
?
Agradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad8^gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*E
_class;
97loc:@gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad
?
,gradients/main/pi/dense_2/MatMul_grad/MatMulMatMul?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencymain/pi/dense_2/kernel/read*(
_output_shapes
:??????????*
transpose_a( *
T0*
transpose_b(
?
.gradients/main/pi/dense_2/MatMul_grad/MatMul_1MatMulmain/pi/dense_1/Relu?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes
:	?*
transpose_b( 
?
6gradients/main/pi/dense_2/MatMul_grad/tuple/group_depsNoOp-^gradients/main/pi/dense_2/MatMul_grad/MatMul/^gradients/main/pi/dense_2/MatMul_grad/MatMul_1
?
>gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_2/MatMul_grad/MatMul7^gradients/main/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
@gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/pi/dense_2/MatMul_grad/MatMul_17^gradients/main/pi/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	?*
T0*A
_class7
53loc:@gradients/main/pi/dense_2/MatMul_grad/MatMul_1
?
,gradients/main/pi/dense_1/Relu_grad/ReluGradReluGrad>gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependencymain/pi/dense_1/Relu*(
_output_shapes
:??????????*
T0
?
2gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/main/pi/dense_1/Relu_grad/ReluGrad*
_output_shapes	
:?*
T0*
data_formatNHWC
?
7gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad-^gradients/main/pi/dense_1/Relu_grad/ReluGrad
?
?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_1/Relu_grad/ReluGrad8^gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:??????????
?
Agradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad8^gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:?*E
_class;
97loc:@gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad
?
,gradients/main/pi/dense_1/MatMul_grad/MatMulMatMul?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencymain/pi/dense_1/kernel/read*(
_output_shapes
:??????????*
transpose_a( *
transpose_b(*
T0
?
.gradients/main/pi/dense_1/MatMul_grad/MatMul_1MatMulmain/pi/dense/Relu?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
??*
T0*
transpose_b( *
transpose_a(
?
6gradients/main/pi/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients/main/pi/dense_1/MatMul_grad/MatMul/^gradients/main/pi/dense_1/MatMul_grad/MatMul_1
?
>gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_1/MatMul_grad/MatMul7^gradients/main/pi/dense_1/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients/main/pi/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
@gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/pi/dense_1/MatMul_grad/MatMul_17^gradients/main/pi/dense_1/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients/main/pi/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
??*
T0
?
*gradients/main/pi/dense/Relu_grad/ReluGradReluGrad>gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependencymain/pi/dense/Relu*(
_output_shapes
:??????????*
T0
?
0gradients/main/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/main/pi/dense/Relu_grad/ReluGrad*
_output_shapes	
:?*
data_formatNHWC*
T0
?
5gradients/main/pi/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad+^gradients/main/pi/dense/Relu_grad/ReluGrad
?
=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity*gradients/main/pi/dense/Relu_grad/ReluGrad6^gradients/main/pi/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:??????????*
T0*=
_class3
1/loc:@gradients/main/pi/dense/Relu_grad/ReluGrad
?
?gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad6^gradients/main/pi/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:?*
T0*C
_class9
75loc:@gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad
?
*gradients/main/pi/dense/MatMul_grad/MatMulMatMul=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependencymain/pi/dense/kernel/read*
transpose_b(*
transpose_a( *'
_output_shapes
:?????????o*
T0
?
,gradients/main/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	o?*
transpose_b( *
T0*
transpose_a(
?
4gradients/main/pi/dense/MatMul_grad/tuple/group_depsNoOp+^gradients/main/pi/dense/MatMul_grad/MatMul-^gradients/main/pi/dense/MatMul_grad/MatMul_1
?
<gradients/main/pi/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients/main/pi/dense/MatMul_grad/MatMul5^gradients/main/pi/dense/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/main/pi/dense/MatMul_grad/MatMul*'
_output_shapes
:?????????o
?
>gradients/main/pi/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients/main/pi/dense/MatMul_grad/MatMul_15^gradients/main/pi/dense/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients/main/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	o?*
T0
?
beta1_power/initial_valueConst*%
_class
loc:@main/pi/dense/bias*
valueB
 *fff?*
_output_shapes
: *
dtype0
?
beta1_power
VariableV2*
dtype0*
shared_name *
shape: *%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
	container 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*%
_class
loc:@main/pi/dense/bias
q
beta1_power/readIdentitybeta1_power*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
?
beta2_power/initial_valueConst*
dtype0*
valueB
 *w??*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
?
beta2_power
VariableV2*
_output_shapes
: *
	container *
shape: *%
_class
loc:@main/pi/dense/bias*
shared_name *
dtype0
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
T0*%
_class
loc:@main/pi/dense/bias*
use_locking(
q
beta2_power/readIdentitybeta2_power*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
?
;main/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"o      *
dtype0*
_output_shapes
:*'
_class
loc:@main/pi/dense/kernel
?
1main/pi/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *'
_class
loc:@main/pi/dense/kernel*
valueB
 *    
?
+main/pi/dense/kernel/Adam/Initializer/zerosFill;main/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1main/pi/dense/kernel/Adam/Initializer/zeros/Const*
T0*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*

index_type0
?
main/pi/dense/kernel/Adam
VariableV2*
shape:	o?*
_output_shapes
:	o?*
	container *
shared_name *'
_class
loc:@main/pi/dense/kernel*
dtype0
?
 main/pi/dense/kernel/Adam/AssignAssignmain/pi/dense/kernel/Adam+main/pi/dense/kernel/Adam/Initializer/zeros*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0*
validate_shape(*
use_locking(
?
main/pi/dense/kernel/Adam/readIdentitymain/pi/dense/kernel/Adam*
T0*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel
?
=main/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"o      *'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:
?
3main/pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
: 
?
-main/pi/dense/kernel/Adam_1/Initializer/zerosFill=main/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3main/pi/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*

index_type0
?
main/pi/dense/kernel/Adam_1
VariableV2*
shared_name *
dtype0*
	container *'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
shape:	o?
?
"main/pi/dense/kernel/Adam_1/AssignAssignmain/pi/dense/kernel/Adam_1-main/pi/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0*
use_locking(
?
 main/pi/dense/kernel/Adam_1/readIdentitymain/pi/dense/kernel/Adam_1*'
_class
loc:@main/pi/dense/kernel*
T0*
_output_shapes
:	o?
?
)main/pi/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
valueB?*    
?
main/pi/dense/bias/Adam
VariableV2*
	container *
shared_name *%
_class
loc:@main/pi/dense/bias*
shape:?*
dtype0*
_output_shapes	
:?
?
main/pi/dense/bias/Adam/AssignAssignmain/pi/dense/bias/Adam)main/pi/dense/bias/Adam/Initializer/zeros*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0
?
main/pi/dense/bias/Adam/readIdentitymain/pi/dense/bias/Adam*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
T0
?
+main/pi/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *%
_class
loc:@main/pi/dense/bias*
dtype0
?
main/pi/dense/bias/Adam_1
VariableV2*
dtype0*
shared_name *
	container *%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?*
shape:?
?
 main/pi/dense/bias/Adam_1/AssignAssignmain/pi/dense/bias/Adam_1+main/pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
T0
?
main/pi/dense/bias/Adam_1/readIdentitymain/pi/dense/bias/Adam_1*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes	
:?
?
=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      *)
_class
loc:@main/pi/dense_1/kernel
?
3main/pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
: 
?
-main/pi/dense_1/kernel/Adam/Initializer/zerosFill=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/pi/dense_1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
??*

index_type0*)
_class
loc:@main/pi/dense_1/kernel*
T0
?
main/pi/dense_1/kernel/Adam
VariableV2* 
_output_shapes
:
??*
shape:
??*
shared_name *)
_class
loc:@main/pi/dense_1/kernel*
	container *
dtype0
?
"main/pi/dense_1/kernel/Adam/AssignAssignmain/pi/dense_1/kernel/Adam-main/pi/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
T0
?
 main/pi/dense_1/kernel/Adam/readIdentitymain/pi/dense_1/kernel/Adam*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
T0
?
?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
:*
valueB"      *
dtype0
?
5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *)
_class
loc:@main/pi/dense_1/kernel
?
/main/pi/dense_1/kernel/Adam_1/Initializer/zerosFill?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_output_shapes
:
??*

index_type0*)
_class
loc:@main/pi/dense_1/kernel
?
main/pi/dense_1/kernel/Adam_1
VariableV2*
shared_name * 
_output_shapes
:
??*
	container *
shape:
??*)
_class
loc:@main/pi/dense_1/kernel*
dtype0
?
$main/pi/dense_1/kernel/Adam_1/AssignAssignmain/pi/dense_1/kernel/Adam_1/main/pi/dense_1/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(*
T0*
validate_shape(
?
"main/pi/dense_1/kernel/Adam_1/readIdentitymain/pi/dense_1/kernel/Adam_1* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
T0
?
+main/pi/dense_1/bias/Adam/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
dtype0
?
main/pi/dense_1/bias/Adam
VariableV2*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias
?
 main/pi/dense_1/bias/Adam/AssignAssignmain/pi/dense_1/bias/Adam+main/pi/dense_1/bias/Adam/Initializer/zeros*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
main/pi/dense_1/bias/Adam/readIdentitymain/pi/dense_1/bias/Adam*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:?
?
-main/pi/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
dtype0
?
main/pi/dense_1/bias/Adam_1
VariableV2*'
_class
loc:@main/pi/dense_1/bias*
	container *
_output_shapes	
:?*
dtype0*
shared_name *
shape:?
?
"main/pi/dense_1/bias/Adam_1/AssignAssignmain/pi/dense_1/bias/Adam_1-main/pi/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*'
_class
loc:@main/pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
 main/pi/dense_1/bias/Adam_1/readIdentitymain/pi/dense_1/bias/Adam_1*
T0*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias
?
=main/pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
dtype0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:
?
3main/pi/dense_2/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*)
_class
loc:@main/pi/dense_2/kernel*
valueB
 *    *
_output_shapes
: 
?
-main/pi/dense_2/kernel/Adam/Initializer/zerosFill=main/pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor3main/pi/dense_2/kernel/Adam/Initializer/zeros/Const*)
_class
loc:@main/pi/dense_2/kernel*

index_type0*
T0*
_output_shapes
:	?
?
main/pi/dense_2/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
shape:	?*
dtype0*
	container 
?
"main/pi/dense_2/kernel/Adam/AssignAssignmain/pi/dense_2/kernel/Adam-main/pi/dense_2/kernel/Adam/Initializer/zeros*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(
?
 main/pi/dense_2/kernel/Adam/readIdentitymain/pi/dense_2/kernel/Adam*)
_class
loc:@main/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
?main/pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:*
valueB"      
?
5main/pi/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@main/pi/dense_2/kernel*
valueB
 *    *
_output_shapes
: *
dtype0
?
/main/pi/dense_2/kernel/Adam_1/Initializer/zerosFill?main/pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/pi/dense_2/kernel/Adam_1/Initializer/zeros/Const*)
_class
loc:@main/pi/dense_2/kernel*

index_type0*
_output_shapes
:	?*
T0
?
main/pi/dense_2/kernel/Adam_1
VariableV2*
shape:	?*)
_class
loc:@main/pi/dense_2/kernel*
shared_name *
	container *
dtype0*
_output_shapes
:	?
?
$main/pi/dense_2/kernel/Adam_1/AssignAssignmain/pi/dense_2/kernel/Adam_1/main/pi/dense_2/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
_output_shapes
:	?*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel
?
"main/pi/dense_2/kernel/Adam_1/readIdentitymain/pi/dense_2/kernel/Adam_1*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
T0
?
+main/pi/dense_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense_2/bias*
dtype0*
_output_shapes
:*
valueB*    
?
main/pi/dense_2/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
shape:*
	container *'
_class
loc:@main/pi/dense_2/bias
?
 main/pi/dense_2/bias/Adam/AssignAssignmain/pi/dense_2/bias/Adam+main/pi/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
T0
?
main/pi/dense_2/bias/Adam/readIdentitymain/pi/dense_2/bias/Adam*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
T0
?
-main/pi/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
valueB*    
?
main/pi/dense_2/bias/Adam_1
VariableV2*
dtype0*
shape:*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
	container *
shared_name 
?
"main/pi/dense_2/bias/Adam_1/AssignAssignmain/pi/dense_2/bias/Adam_1-main/pi/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
?
 main/pi/dense_2/bias/Adam_1/readIdentitymain/pi/dense_2/bias/Adam_1*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
T0
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *o?:*
dtype0
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *w??*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *w?+2*
dtype0
?
*Adam/update_main/pi/dense/kernel/ApplyAdam	ApplyAdammain/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/pi/dense/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
use_locking( 
?
(Adam/update_main/pi/dense/bias/ApplyAdam	ApplyAdammain/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *%
_class
loc:@main/pi/dense/bias*
T0*
use_locking( *
_output_shapes	
:?
?
,Adam/update_main/pi/dense_1/kernel/ApplyAdam	ApplyAdammain/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*)
_class
loc:@main/pi/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
??*
use_locking( 
?
*Adam/update_main/pi/dense_1/bias/ApplyAdam	ApplyAdammain/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?
?
,Adam/update_main/pi/dense_2/kernel/ApplyAdam	ApplyAdammain/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
use_nesterov( *)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?
?
*Adam/update_main/pi/dense_2/bias/ApplyAdam	ApplyAdammain/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
T0*
use_nesterov( *'
_class
loc:@main/pi/dense_2/bias*
use_locking( 
?
Adam/mulMulbeta1_power/read
Adam/beta1)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
?
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
use_locking( *
validate_shape(*
T0
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
T0
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
validate_shape(*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
?
AdamNoOp^Adam/Assign^Adam/Assign_1)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam
T
gradients_1/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*
_output_shapes
: *

index_type0
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
`
gradients_1/Mean_1_grad/ShapeShapepow*
out_type0*
_output_shapes
:*
T0
?
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*#
_output_shapes
:?????????*

Tmultiples0*
T0
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
out_type0*
_output_shapes
:*
T0
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
?
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
?
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
?
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
?
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
?
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:?????????
_
gradients_1/pow_grad/ShapeShapesub_1*
T0*
_output_shapes
:*
out_type0
_
gradients_1/pow_grad/Shape_1Shapepow/y*
_output_shapes
: *
T0*
out_type0
?
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*#
_output_shapes
:?????????*
T0
_
gradients_1/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*
T0*#
_output_shapes
:?????????
?
gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*
T0*#
_output_shapes
:?????????
?
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*
Tshape0*#
_output_shapes
:?????????*
T0
c
gradients_1/pow_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*#
_output_shapes
:?????????*
T0
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
_output_shapes
:*
out_type0*
T0
i
$gradients_1/pow_grad/ones_like/ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*#
_output_shapes
:?????????*
T0*

index_type0
?
gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*
T0*#
_output_shapes
:?????????
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*#
_output_shapes
:?????????*
T0
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*
T0*#
_output_shapes
:?????????
?
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*
T0*#
_output_shapes
:?????????
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*#
_output_shapes
:?????????*
T0
?
gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*
T0*#
_output_shapes
:?????????
?
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
?
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*#
_output_shapes
:?????????
?
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1
j
gradients_1/sub_1_grad/ShapeShapemain/q/Squeeze*
T0*
_output_shapes
:*
out_type0
j
gradients_1/sub_1_grad/Shape_1ShapeStopGradient*
_output_shapes
:*
T0*
out_type0
?
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
~
gradients_1/sub_1_grad/NegNeg-gradients_1/pow_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
gradients_1/sub_1_grad/Sum_1Sumgradients_1/sub_1_grad/Neg.gradients_1/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Sum_1gradients_1/sub_1_grad/Shape_1*
T0*#
_output_shapes
:?????????*
Tshape0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
?
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
T0*#
_output_shapes
:?????????
?
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*#
_output_shapes
:?????????*
T0
{
%gradients_1/main/q/Squeeze_grad/ShapeShapemain/q/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
?
'gradients_1/main/q/Squeeze_grad/ReshapeReshape/gradients_1/sub_1_grad/tuple/control_dependency%gradients_1/main/q/Squeeze_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????
?
3gradients_1/main/q/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/main/q/Squeeze_grad/Reshape*
_output_shapes
:*
data_formatNHWC*
T0
?
8gradients_1/main/q/dense_2/BiasAdd_grad/tuple/group_depsNoOp(^gradients_1/main/q/Squeeze_grad/Reshape4^gradients_1/main/q/dense_2/BiasAdd_grad/BiasAddGrad
?
@gradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_1/main/q/Squeeze_grad/Reshape9^gradients_1/main/q/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*:
_class0
.,loc:@gradients_1/main/q/Squeeze_grad/Reshape
?
Bgradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_1/main/q/dense_2/BiasAdd_grad/BiasAddGrad9^gradients_1/main/q/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*F
_class<
:8loc:@gradients_1/main/q/dense_2/BiasAdd_grad/BiasAddGrad
?
-gradients_1/main/q/dense_2/MatMul_grad/MatMulMatMul@gradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependencymain/q/dense_2/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_b(*
transpose_a( 
?
/gradients_1/main/q/dense_2/MatMul_grad/MatMul_1MatMulmain/q/dense_1/Relu@gradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	?
?
7gradients_1/main/q/dense_2/MatMul_grad/tuple/group_depsNoOp.^gradients_1/main/q/dense_2/MatMul_grad/MatMul0^gradients_1/main/q/dense_2/MatMul_grad/MatMul_1
?
?gradients_1/main/q/dense_2/MatMul_grad/tuple/control_dependencyIdentity-gradients_1/main/q/dense_2/MatMul_grad/MatMul8^gradients_1/main/q/dense_2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/main/q/dense_2/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
Agradients_1/main/q/dense_2/MatMul_grad/tuple/control_dependency_1Identity/gradients_1/main/q/dense_2/MatMul_grad/MatMul_18^gradients_1/main/q/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	?*B
_class8
64loc:@gradients_1/main/q/dense_2/MatMul_grad/MatMul_1
?
-gradients_1/main/q/dense_1/Relu_grad/ReluGradReluGrad?gradients_1/main/q/dense_2/MatMul_grad/tuple/control_dependencymain/q/dense_1/Relu*(
_output_shapes
:??????????*
T0
?
3gradients_1/main/q/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients_1/main/q/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:?
?
8gradients_1/main/q/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^gradients_1/main/q/dense_1/BiasAdd_grad/BiasAddGrad.^gradients_1/main/q/dense_1/Relu_grad/ReluGrad
?
@gradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-gradients_1/main/q/dense_1/Relu_grad/ReluGrad9^gradients_1/main/q/dense_1/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:??????????*@
_class6
42loc:@gradients_1/main/q/dense_1/Relu_grad/ReluGrad
?
Bgradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_1/main/q/dense_1/BiasAdd_grad/BiasAddGrad9^gradients_1/main/q/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:?*
T0*F
_class<
:8loc:@gradients_1/main/q/dense_1/BiasAdd_grad/BiasAddGrad
?
-gradients_1/main/q/dense_1/MatMul_grad/MatMulMatMul@gradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependencymain/q/dense_1/kernel/read*
transpose_a( *(
_output_shapes
:??????????*
transpose_b(*
T0
?
/gradients_1/main/q/dense_1/MatMul_grad/MatMul_1MatMulmain/q/dense/Relu@gradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a(
?
7gradients_1/main/q/dense_1/MatMul_grad/tuple/group_depsNoOp.^gradients_1/main/q/dense_1/MatMul_grad/MatMul0^gradients_1/main/q/dense_1/MatMul_grad/MatMul_1
?
?gradients_1/main/q/dense_1/MatMul_grad/tuple/control_dependencyIdentity-gradients_1/main/q/dense_1/MatMul_grad/MatMul8^gradients_1/main/q/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:??????????*@
_class6
42loc:@gradients_1/main/q/dense_1/MatMul_grad/MatMul*
T0
?
Agradients_1/main/q/dense_1/MatMul_grad/tuple/control_dependency_1Identity/gradients_1/main/q/dense_1/MatMul_grad/MatMul_18^gradients_1/main/q/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/main/q/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
??
?
+gradients_1/main/q/dense/Relu_grad/ReluGradReluGrad?gradients_1/main/q/dense_1/MatMul_grad/tuple/control_dependencymain/q/dense/Relu*
T0*(
_output_shapes
:??????????
?
1gradients_1/main/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients_1/main/q/dense/Relu_grad/ReluGrad*
_output_shapes	
:?*
T0*
data_formatNHWC
?
6gradients_1/main/q/dense/BiasAdd_grad/tuple/group_depsNoOp2^gradients_1/main/q/dense/BiasAdd_grad/BiasAddGrad,^gradients_1/main/q/dense/Relu_grad/ReluGrad
?
>gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependencyIdentity+gradients_1/main/q/dense/Relu_grad/ReluGrad7^gradients_1/main/q/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:??????????*
T0*>
_class4
20loc:@gradients_1/main/q/dense/Relu_grad/ReluGrad
?
@gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependency_1Identity1gradients_1/main/q/dense/BiasAdd_grad/BiasAddGrad7^gradients_1/main/q/dense/BiasAdd_grad/tuple/group_deps*D
_class:
86loc:@gradients_1/main/q/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:?
?
+gradients_1/main/q/dense/MatMul_grad/MatMulMatMul>gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependencymain/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:?????????w*
transpose_b(
?
-gradients_1/main/q/dense/MatMul_grad/MatMul_1MatMulmain/q/concat>gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	w?*
transpose_b( *
T0*
transpose_a(
?
5gradients_1/main/q/dense/MatMul_grad/tuple/group_depsNoOp,^gradients_1/main/q/dense/MatMul_grad/MatMul.^gradients_1/main/q/dense/MatMul_grad/MatMul_1
?
=gradients_1/main/q/dense/MatMul_grad/tuple/control_dependencyIdentity+gradients_1/main/q/dense/MatMul_grad/MatMul6^gradients_1/main/q/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????w*>
_class4
20loc:@gradients_1/main/q/dense/MatMul_grad/MatMul*
T0
?
?gradients_1/main/q/dense/MatMul_grad/tuple/control_dependency_1Identity-gradients_1/main/q/dense/MatMul_grad/MatMul_16^gradients_1/main/q/dense/MatMul_grad/tuple/group_deps*@
_class6
42loc:@gradients_1/main/q/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	w?
?
beta1_power_1/initial_valueConst*$
_class
loc:@main/q/dense/bias*
dtype0*
_output_shapes
: *
valueB
 *fff?
?
beta1_power_1
VariableV2*
dtype0*
shape: *
_output_shapes
: *$
_class
loc:@main/q/dense/bias*
shared_name *
	container 
?
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*$
_class
loc:@main/q/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
t
beta1_power_1/readIdentitybeta1_power_1*$
_class
loc:@main/q/dense/bias*
_output_shapes
: *
T0
?
beta2_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *w??*
dtype0*$
_class
loc:@main/q/dense/bias
?
beta2_power_1
VariableV2*
shape: *
_output_shapes
: *
dtype0*$
_class
loc:@main/q/dense/bias*
	container *
shared_name 
?
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
t
beta2_power_1/readIdentitybeta2_power_1*
_output_shapes
: *$
_class
loc:@main/q/dense/bias*
T0
?
:main/q/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"w      *
_output_shapes
:*&
_class
loc:@main/q/dense/kernel
?
0main/q/dense/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@main/q/dense/kernel*
dtype0*
_output_shapes
: *
valueB
 *    
?
*main/q/dense/kernel/Adam/Initializer/zerosFill:main/q/dense/kernel/Adam/Initializer/zeros/shape_as_tensor0main/q/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	w?*
T0*&
_class
loc:@main/q/dense/kernel*

index_type0
?
main/q/dense/kernel/Adam
VariableV2*
dtype0*&
_class
loc:@main/q/dense/kernel*
	container *
shared_name *
shape:	w?*
_output_shapes
:	w?
?
main/q/dense/kernel/Adam/AssignAssignmain/q/dense/kernel/Adam*main/q/dense/kernel/Adam/Initializer/zeros*
_output_shapes
:	w?*
use_locking(*
T0*
validate_shape(*&
_class
loc:@main/q/dense/kernel
?
main/q/dense/kernel/Adam/readIdentitymain/q/dense/kernel/Adam*
_output_shapes
:	w?*
T0*&
_class
loc:@main/q/dense/kernel
?
<main/q/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:*
valueB"w      *
dtype0
?
2main/q/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *&
_class
loc:@main/q/dense/kernel*
valueB
 *    *
dtype0
?
,main/q/dense/kernel/Adam_1/Initializer/zerosFill<main/q/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/q/dense/kernel/Adam_1/Initializer/zeros/Const*

index_type0*
_output_shapes
:	w?*
T0*&
_class
loc:@main/q/dense/kernel
?
main/q/dense/kernel/Adam_1
VariableV2*
_output_shapes
:	w?*
dtype0*&
_class
loc:@main/q/dense/kernel*
shape:	w?*
shared_name *
	container 
?
!main/q/dense/kernel/Adam_1/AssignAssignmain/q/dense/kernel/Adam_1,main/q/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	w?*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
T0*
use_locking(
?
main/q/dense/kernel/Adam_1/readIdentitymain/q/dense/kernel/Adam_1*&
_class
loc:@main/q/dense/kernel*
T0*
_output_shapes
:	w?
?
(main/q/dense/bias/Adam/Initializer/zerosConst*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
dtype0*
valueB?*    
?
main/q/dense/bias/Adam
VariableV2*
shared_name *
_output_shapes	
:?*
dtype0*
	container *$
_class
loc:@main/q/dense/bias*
shape:?
?
main/q/dense/bias/Adam/AssignAssignmain/q/dense/bias/Adam(main/q/dense/bias/Adam/Initializer/zeros*$
_class
loc:@main/q/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?
?
main/q/dense/bias/Adam/readIdentitymain/q/dense/bias/Adam*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes	
:?
?
*main/q/dense/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?*
valueB?*    *
dtype0
?
main/q/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *$
_class
loc:@main/q/dense/bias*
shape:?*
	container 
?
main/q/dense/bias/Adam_1/AssignAssignmain/q/dense/bias/Adam_1*main/q/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias
?
main/q/dense/bias/Adam_1/readIdentitymain/q/dense/bias/Adam_1*
_output_shapes	
:?*
T0*$
_class
loc:@main/q/dense/bias
?
<main/q/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:*(
_class
loc:@main/q/dense_1/kernel
?
2main/q/dense_1/kernel/Adam/Initializer/zeros/ConstConst*(
_class
loc:@main/q/dense_1/kernel*
dtype0*
valueB
 *    *
_output_shapes
: 
?
,main/q/dense_1/kernel/Adam/Initializer/zerosFill<main/q/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor2main/q/dense_1/kernel/Adam/Initializer/zeros/Const*
T0* 
_output_shapes
:
??*

index_type0*(
_class
loc:@main/q/dense_1/kernel
?
main/q/dense_1/kernel/Adam
VariableV2* 
_output_shapes
:
??*
shape:
??*
shared_name *
dtype0*
	container *(
_class
loc:@main/q/dense_1/kernel
?
!main/q/dense_1/kernel/Adam/AssignAssignmain/q/dense_1/kernel/Adam,main/q/dense_1/kernel/Adam/Initializer/zeros* 
_output_shapes
:
??*
T0*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel*
use_locking(
?
main/q/dense_1/kernel/Adam/readIdentitymain/q/dense_1/kernel/Adam* 
_output_shapes
:
??*
T0*(
_class
loc:@main/q/dense_1/kernel
?
>main/q/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      *(
_class
loc:@main/q/dense_1/kernel
?
4main/q/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*(
_class
loc:@main/q/dense_1/kernel*
valueB
 *    
?
.main/q/dense_1/kernel/Adam_1/Initializer/zerosFill>main/q/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor4main/q/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
??*
T0*(
_class
loc:@main/q/dense_1/kernel*

index_type0
?
main/q/dense_1/kernel/Adam_1
VariableV2*
	container *(
_class
loc:@main/q/dense_1/kernel*
shape:
??*
dtype0*
shared_name * 
_output_shapes
:
??
?
#main/q/dense_1/kernel/Adam_1/AssignAssignmain/q/dense_1/kernel/Adam_1.main/q/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
use_locking(*
T0
?
!main/q/dense_1/kernel/Adam_1/readIdentitymain/q/dense_1/kernel/Adam_1*(
_class
loc:@main/q/dense_1/kernel*
T0* 
_output_shapes
:
??
?
*main/q/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
valueB?*    
?
main/q/dense_1/bias/Adam
VariableV2*
shared_name *
dtype0*&
_class
loc:@main/q/dense_1/bias*
	container *
_output_shapes	
:?*
shape:?
?
main/q/dense_1/bias/Adam/AssignAssignmain/q/dense_1/bias/Adam*main/q/dense_1/bias/Adam/Initializer/zeros*
T0*
use_locking(*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:?
?
main/q/dense_1/bias/Adam/readIdentitymain/q/dense_1/bias/Adam*
T0*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias
?
,main/q/dense_1/bias/Adam_1/Initializer/zerosConst*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
valueB?*    *
dtype0
?
main/q/dense_1/bias/Adam_1
VariableV2*
	container *
_output_shapes	
:?*
shape:?*
dtype0*
shared_name *&
_class
loc:@main/q/dense_1/bias
?
!main/q/dense_1/bias/Adam_1/AssignAssignmain/q/dense_1/bias/Adam_1,main/q/dense_1/bias/Adam_1/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*&
_class
loc:@main/q/dense_1/bias
?
main/q/dense_1/bias/Adam_1/readIdentitymain/q/dense_1/bias/Adam_1*
T0*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias
?
,main/q/dense_2/kernel/Adam/Initializer/zerosConst*
_output_shapes
:	?*
dtype0*(
_class
loc:@main/q/dense_2/kernel*
valueB	?*    
?
main/q/dense_2/kernel/Adam
VariableV2*
_output_shapes
:	?*
shared_name *
	container *(
_class
loc:@main/q/dense_2/kernel*
dtype0*
shape:	?
?
!main/q/dense_2/kernel/Adam/AssignAssignmain/q/dense_2/kernel/Adam,main/q/dense_2/kernel/Adam/Initializer/zeros*
T0*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	?
?
main/q/dense_2/kernel/Adam/readIdentitymain/q/dense_2/kernel/Adam*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
T0
?
.main/q/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes
:	?*
valueB	?*    *
dtype0*(
_class
loc:@main/q/dense_2/kernel
?
main/q/dense_2/kernel/Adam_1
VariableV2*
shape:	?*
shared_name *
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
	container *
dtype0
?
#main/q/dense_2/kernel/Adam_1/AssignAssignmain/q/dense_2/kernel/Adam_1.main/q/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
T0*
_output_shapes
:	?
?
!main/q/dense_2/kernel/Adam_1/readIdentitymain/q/dense_2/kernel/Adam_1*(
_class
loc:@main/q/dense_2/kernel*
T0*
_output_shapes
:	?
?
*main/q/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
valueB*    
?
main/q/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
dtype0*
shape:*&
_class
loc:@main/q/dense_2/bias*
	container *
shared_name 
?
main/q/dense_2/bias/Adam/AssignAssignmain/q/dense_2/bias/Adam*main/q/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*&
_class
loc:@main/q/dense_2/bias
?
main/q/dense_2/bias/Adam/readIdentitymain/q/dense_2/bias/Adam*
T0*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias
?
,main/q/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias*
valueB*    *
dtype0
?
main/q/dense_2/bias/Adam_1
VariableV2*
	container *&
_class
loc:@main/q/dense_2/bias*
shared_name *
shape:*
_output_shapes
:*
dtype0
?
!main/q/dense_2/bias/Adam_1/AssignAssignmain/q/dense_2/bias/Adam_1,main/q/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias*
T0*
validate_shape(
?
main/q/dense_2/bias/Adam_1/readIdentitymain/q/dense_2/bias/Adam_1*
_output_shapes
:*
T0*&
_class
loc:@main/q/dense_2/bias
Y
Adam_1/learning_rateConst*
dtype0*
valueB
 *RI?9*
_output_shapes
: 
Q
Adam_1/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
Q
Adam_1/beta2Const*
_output_shapes
: *
valueB
 *w??*
dtype0
S
Adam_1/epsilonConst*
valueB
 *w?+2*
_output_shapes
: *
dtype0
?
+Adam_1/update_main/q/dense/kernel/ApplyAdam	ApplyAdammain/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon?gradients_1/main/q/dense/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
use_locking( *
T0
?
)Adam_1/update_main/q/dense/bias/ApplyAdam	ApplyAdammain/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon@gradients_1/main/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?*
T0
?
-Adam_1/update_main/q/dense_1/kernel/ApplyAdam	ApplyAdammain/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/q/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *(
_class
loc:@main/q/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
??
?
+Adam_1/update_main/q/dense_1/bias/ApplyAdam	ApplyAdammain/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q/dense_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
use_locking( *
T0*
use_nesterov( 
?
-Adam_1/update_main/q/dense_2/kernel/ApplyAdam	ApplyAdammain/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/q/dense_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	?*
use_locking( *
T0
?
+Adam_1/update_main/q/dense_2/bias/ApplyAdam	ApplyAdammain/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:*
use_nesterov( *&
_class
loc:@main/q/dense_2/bias*
use_locking( 
?

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1*^Adam_1/update_main/q/dense/bias/ApplyAdam,^Adam_1/update_main/q/dense/kernel/ApplyAdam,^Adam_1/update_main/q/dense_1/bias/ApplyAdam.^Adam_1/update_main/q/dense_1/kernel/ApplyAdam,^Adam_1/update_main/q/dense_2/bias/ApplyAdam.^Adam_1/update_main/q/dense_2/kernel/ApplyAdam*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes
: 
?
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*$
_class
loc:@main/q/dense/bias*
use_locking( *
validate_shape(*
_output_shapes
: 
?
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2*^Adam_1/update_main/q/dense/bias/ApplyAdam,^Adam_1/update_main/q/dense/kernel/ApplyAdam,^Adam_1/update_main/q/dense_1/bias/ApplyAdam.^Adam_1/update_main/q/dense_1/kernel/ApplyAdam,^Adam_1/update_main/q/dense_2/bias/ApplyAdam.^Adam_1/update_main/q/dense_2/kernel/ApplyAdam*
T0*
_output_shapes
: *$
_class
loc:@main/q/dense/bias
?
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
_output_shapes
: *
use_locking( *
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias
?
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1*^Adam_1/update_main/q/dense/bias/ApplyAdam,^Adam_1/update_main/q/dense/kernel/ApplyAdam,^Adam_1/update_main/q/dense_1/bias/ApplyAdam.^Adam_1/update_main/q/dense_1/kernel/ApplyAdam,^Adam_1/update_main/q/dense_2/bias/ApplyAdam.^Adam_1/update_main/q/dense_2/kernel/ApplyAdam
L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *R?~?
\
mul_2Mulmul_2/xtarget/pi/dense/kernel/read*
_output_shapes
:	o?*
T0
L
mul_3/xConst*
valueB
 *
ף;*
_output_shapes
: *
dtype0
Z
mul_3Mulmul_3/xmain/pi/dense/kernel/read*
T0*
_output_shapes
:	o?
F
add_1AddV2mul_2mul_3*
T0*
_output_shapes
:	o?
?
AssignAssigntarget/pi/dense/kerneladd_1*
validate_shape(*
use_locking(*
_output_shapes
:	o?*)
_class
loc:@target/pi/dense/kernel*
T0
L
mul_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *R?~?
V
mul_4Mulmul_4/xtarget/pi/dense/bias/read*
T0*
_output_shapes	
:?
L
mul_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *
ף;
T
mul_5Mulmul_5/xmain/pi/dense/bias/read*
_output_shapes	
:?*
T0
B
add_2AddV2mul_4mul_5*
_output_shapes	
:?*
T0
?
Assign_1Assigntarget/pi/dense/biasadd_2*
validate_shape(*'
_class
loc:@target/pi/dense/bias*
T0*
_output_shapes	
:?*
use_locking(
L
mul_6/xConst*
dtype0*
valueB
 *R?~?*
_output_shapes
: 
_
mul_6Mulmul_6/xtarget/pi/dense_1/kernel/read*
T0* 
_output_shapes
:
??
L
mul_7/xConst*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
]
mul_7Mulmul_7/xmain/pi/dense_1/kernel/read*
T0* 
_output_shapes
:
??
G
add_3AddV2mul_6mul_7* 
_output_shapes
:
??*
T0
?
Assign_2Assigntarget/pi/dense_1/kerneladd_3*
validate_shape(*
T0* 
_output_shapes
:
??*+
_class!
loc:@target/pi/dense_1/kernel*
use_locking(
L
mul_8/xConst*
dtype0*
_output_shapes
: *
valueB
 *R?~?
X
mul_8Mulmul_8/xtarget/pi/dense_1/bias/read*
_output_shapes	
:?*
T0
L
mul_9/xConst*
_output_shapes
: *
valueB
 *
ף;*
dtype0
V
mul_9Mulmul_9/xmain/pi/dense_1/bias/read*
T0*
_output_shapes	
:?
B
add_4AddV2mul_8mul_9*
_output_shapes	
:?*
T0
?
Assign_3Assigntarget/pi/dense_1/biasadd_4*
_output_shapes	
:?*
validate_shape(*)
_class
loc:@target/pi/dense_1/bias*
use_locking(*
T0
M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *R?~?
`
mul_10Mulmul_10/xtarget/pi/dense_2/kernel/read*
T0*
_output_shapes
:	?
M
mul_11/xConst*
_output_shapes
: *
valueB
 *
ף;*
dtype0
^
mul_11Mulmul_11/xmain/pi/dense_2/kernel/read*
T0*
_output_shapes
:	?
H
add_5AddV2mul_10mul_11*
_output_shapes
:	?*
T0
?
Assign_4Assigntarget/pi/dense_2/kerneladd_5*
use_locking(*
validate_shape(*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
_output_shapes
:	?
M
mul_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *R?~?
Y
mul_12Mulmul_12/xtarget/pi/dense_2/bias/read*
_output_shapes
:*
T0
M
mul_13/xConst*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
W
mul_13Mulmul_13/xmain/pi/dense_2/bias/read*
_output_shapes
:*
T0
C
add_6AddV2mul_12mul_13*
_output_shapes
:*
T0
?
Assign_5Assigntarget/pi/dense_2/biasadd_6*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
M
mul_14/xConst*
_output_shapes
: *
valueB
 *R?~?*
dtype0
]
mul_14Mulmul_14/xtarget/q/dense/kernel/read*
T0*
_output_shapes
:	w?
M
mul_15/xConst*
_output_shapes
: *
valueB
 *
ף;*
dtype0
[
mul_15Mulmul_15/xmain/q/dense/kernel/read*
_output_shapes
:	w?*
T0
H
add_7AddV2mul_14mul_15*
T0*
_output_shapes
:	w?
?
Assign_6Assigntarget/q/dense/kerneladd_7*(
_class
loc:@target/q/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	w?
M
mul_16/xConst*
valueB
 *R?~?*
dtype0*
_output_shapes
: 
W
mul_16Mulmul_16/xtarget/q/dense/bias/read*
T0*
_output_shapes	
:?
M
mul_17/xConst*
dtype0*
valueB
 *
ף;*
_output_shapes
: 
U
mul_17Mulmul_17/xmain/q/dense/bias/read*
T0*
_output_shapes	
:?
D
add_8AddV2mul_16mul_17*
T0*
_output_shapes	
:?
?
Assign_7Assigntarget/q/dense/biasadd_8*
_output_shapes	
:?*&
_class
loc:@target/q/dense/bias*
use_locking(*
validate_shape(*
T0
M
mul_18/xConst*
valueB
 *R?~?*
dtype0*
_output_shapes
: 
`
mul_18Mulmul_18/xtarget/q/dense_1/kernel/read* 
_output_shapes
:
??*
T0
M
mul_19/xConst*
valueB
 *
ף;*
_output_shapes
: *
dtype0
^
mul_19Mulmul_19/xmain/q/dense_1/kernel/read* 
_output_shapes
:
??*
T0
I
add_9AddV2mul_18mul_19* 
_output_shapes
:
??*
T0
?
Assign_8Assigntarget/q/dense_1/kerneladd_9*
T0**
_class 
loc:@target/q/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
use_locking(
M
mul_20/xConst*
valueB
 *R?~?*
_output_shapes
: *
dtype0
Y
mul_20Mulmul_20/xtarget/q/dense_1/bias/read*
T0*
_output_shapes	
:?
M
mul_21/xConst*
dtype0*
valueB
 *
ף;*
_output_shapes
: 
W
mul_21Mulmul_21/xmain/q/dense_1/bias/read*
T0*
_output_shapes	
:?
E
add_10AddV2mul_20mul_21*
_output_shapes	
:?*
T0
?
Assign_9Assigntarget/q/dense_1/biasadd_10*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias*
use_locking(*
T0*
validate_shape(
M
mul_22/xConst*
_output_shapes
: *
valueB
 *R?~?*
dtype0
_
mul_22Mulmul_22/xtarget/q/dense_2/kernel/read*
_output_shapes
:	?*
T0
M
mul_23/xConst*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
]
mul_23Mulmul_23/xmain/q/dense_2/kernel/read*
_output_shapes
:	?*
T0
I
add_11AddV2mul_22mul_23*
T0*
_output_shapes
:	?
?
	Assign_10Assigntarget/q/dense_2/kerneladd_11*
validate_shape(*
_output_shapes
:	?*
T0*
use_locking(**
_class 
loc:@target/q/dense_2/kernel
M
mul_24/xConst*
dtype0*
_output_shapes
: *
valueB
 *R?~?
X
mul_24Mulmul_24/xtarget/q/dense_2/bias/read*
T0*
_output_shapes
:
M
mul_25/xConst*
dtype0*
_output_shapes
: *
valueB
 *
ף;
V
mul_25Mulmul_25/xmain/q/dense_2/bias/read*
_output_shapes
:*
T0
D
add_12AddV2mul_24mul_25*
_output_shapes
:*
T0
?
	Assign_11Assigntarget/q/dense_2/biasadd_12*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*(
_class
loc:@target/q/dense_2/bias
?

group_depsNoOp^Assign	^Assign_1
^Assign_10
^Assign_11	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
?
	Assign_12Assigntarget/pi/dense/kernelmain/pi/dense/kernel/read*
use_locking(*
validate_shape(*)
_class
loc:@target/pi/dense/kernel*
T0*
_output_shapes
:	o?
?
	Assign_13Assigntarget/pi/dense/biasmain/pi/dense/bias/read*
use_locking(*
validate_shape(*'
_class
loc:@target/pi/dense/bias*
_output_shapes	
:?*
T0
?
	Assign_14Assigntarget/pi/dense_1/kernelmain/pi/dense_1/kernel/read*
validate_shape(*
T0*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
??*
use_locking(
?
	Assign_15Assigntarget/pi/dense_1/biasmain/pi/dense_1/bias/read*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
	Assign_16Assigntarget/pi/dense_2/kernelmain/pi/dense_2/kernel/read*
T0*
_output_shapes
:	?*
validate_shape(*+
_class!
loc:@target/pi/dense_2/kernel*
use_locking(
?
	Assign_17Assigntarget/pi/dense_2/biasmain/pi/dense_2/bias/read*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias
?
	Assign_18Assigntarget/q/dense/kernelmain/q/dense/kernel/read*
_output_shapes
:	w?*
T0*
use_locking(*(
_class
loc:@target/q/dense/kernel*
validate_shape(
?
	Assign_19Assigntarget/q/dense/biasmain/q/dense/bias/read*&
_class
loc:@target/q/dense/bias*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(
?
	Assign_20Assigntarget/q/dense_1/kernelmain/q/dense_1/kernel/read* 
_output_shapes
:
??**
_class 
loc:@target/q/dense_1/kernel*
validate_shape(*
use_locking(*
T0
?
	Assign_21Assigntarget/q/dense_1/biasmain/q/dense_1/bias/read*
T0*
use_locking(*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
_output_shapes	
:?
?
	Assign_22Assigntarget/q/dense_2/kernelmain/q/dense_2/kernel/read*
validate_shape(*
_output_shapes
:	?**
_class 
loc:@target/q/dense_2/kernel*
use_locking(*
T0
?
	Assign_23Assigntarget/q/dense_2/biasmain/q/dense_2/bias/read*
validate_shape(*
T0*
_output_shapes
:*(
_class
loc:@target/q/dense_2/bias*
use_locking(
?
group_deps_1NoOp
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
?
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^main/pi/dense/bias/Adam/Assign!^main/pi/dense/bias/Adam_1/Assign^main/pi/dense/bias/Assign!^main/pi/dense/kernel/Adam/Assign#^main/pi/dense/kernel/Adam_1/Assign^main/pi/dense/kernel/Assign!^main/pi/dense_1/bias/Adam/Assign#^main/pi/dense_1/bias/Adam_1/Assign^main/pi/dense_1/bias/Assign#^main/pi/dense_1/kernel/Adam/Assign%^main/pi/dense_1/kernel/Adam_1/Assign^main/pi/dense_1/kernel/Assign!^main/pi/dense_2/bias/Adam/Assign#^main/pi/dense_2/bias/Adam_1/Assign^main/pi/dense_2/bias/Assign#^main/pi/dense_2/kernel/Adam/Assign%^main/pi/dense_2/kernel/Adam_1/Assign^main/pi/dense_2/kernel/Assign^main/q/dense/bias/Adam/Assign ^main/q/dense/bias/Adam_1/Assign^main/q/dense/bias/Assign ^main/q/dense/kernel/Adam/Assign"^main/q/dense/kernel/Adam_1/Assign^main/q/dense/kernel/Assign ^main/q/dense_1/bias/Adam/Assign"^main/q/dense_1/bias/Adam_1/Assign^main/q/dense_1/bias/Assign"^main/q/dense_1/kernel/Adam/Assign$^main/q/dense_1/kernel/Adam_1/Assign^main/q/dense_1/kernel/Assign ^main/q/dense_2/bias/Adam/Assign"^main/q/dense_2/bias/Adam_1/Assign^main/q/dense_2/bias/Assign"^main/q/dense_2/kernel/Adam/Assign$^main/q/dense_2/kernel/Adam_1/Assign^main/q/dense_2/kernel/Assign^target/pi/dense/bias/Assign^target/pi/dense/kernel/Assign^target/pi/dense_1/bias/Assign ^target/pi/dense_1/kernel/Assign^target/pi/dense_2/bias/Assign ^target/pi/dense_2/kernel/Assign^target/q/dense/bias/Assign^target/q/dense/kernel/Assign^target/q/dense_1/bias/Assign^target/q/dense_1/kernel/Assign^target/q/dense_2/bias/Assign^target/q/dense_2/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
?
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_eb99a9d1d7444c17b4cca9547a3c3a90/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
?

save/SaveV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
save/SaveV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4*
dtype0
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
_output_shapes
:*
T0*
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
?

save/RestoreV2/tensor_namesConst*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0
?
save/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*B
dtypes8
624*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes
: 
?
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
_output_shapes
: *
validate_shape(*
T0*$
_class
loc:@main/q/dense/bias*
use_locking(
?
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
use_locking(*
T0*
validate_shape(
?
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
?
save/Assign_4Assignmain/pi/dense/biassave/RestoreV2:4*%
_class
loc:@main/pi/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
?
save/Assign_5Assignmain/pi/dense/bias/Adamsave/RestoreV2:5*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_6Assignmain/pi/dense/bias/Adam_1save/RestoreV2:6*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias
?
save/Assign_7Assignmain/pi/dense/kernelsave/RestoreV2:7*
_output_shapes
:	o?*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
T0*
validate_shape(
?
save/Assign_8Assignmain/pi/dense/kernel/Adamsave/RestoreV2:8*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0*
use_locking(*
validate_shape(
?
save/Assign_9Assignmain/pi/dense/kernel/Adam_1save/RestoreV2:9*
T0*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
use_locking(
?
save/Assign_10Assignmain/pi/dense_1/biassave/RestoreV2:10*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:?*
use_locking(
?
save/Assign_11Assignmain/pi/dense_1/bias/Adamsave/RestoreV2:11*
use_locking(*
T0*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?
?
save/Assign_12Assignmain/pi/dense_1/bias/Adam_1save/RestoreV2:12*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias
?
save/Assign_13Assignmain/pi/dense_1/kernelsave/RestoreV2:13*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(* 
_output_shapes
:
??*
T0*
validate_shape(
?
save/Assign_14Assignmain/pi/dense_1/kernel/Adamsave/RestoreV2:14*
T0*
validate_shape(* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(
?
save/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save/RestoreV2:15*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel
?
save/Assign_16Assignmain/pi/dense_2/biassave/RestoreV2:16*
_output_shapes
:*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
use_locking(
?
save/Assign_17Assignmain/pi/dense_2/bias/Adamsave/RestoreV2:17*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
?
save/Assign_18Assignmain/pi/dense_2/bias/Adam_1save/RestoreV2:18*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
?
save/Assign_19Assignmain/pi/dense_2/kernelsave/RestoreV2:19*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
validate_shape(*
T0*
use_locking(
?
save/Assign_20Assignmain/pi/dense_2/kernel/Adamsave/RestoreV2:20*
validate_shape(*
use_locking(*
_output_shapes
:	?*
T0*)
_class
loc:@main/pi/dense_2/kernel
?
save/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save/RestoreV2:21*
validate_shape(*
_output_shapes
:	?*
T0*)
_class
loc:@main/pi/dense_2/kernel*
use_locking(
?
save/Assign_22Assignmain/q/dense/biassave/RestoreV2:22*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0*$
_class
loc:@main/q/dense/bias
?
save/Assign_23Assignmain/q/dense/bias/Adamsave/RestoreV2:23*
validate_shape(*
T0*
use_locking(*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?
?
save/Assign_24Assignmain/q/dense/bias/Adam_1save/RestoreV2:24*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias
?
save/Assign_25Assignmain/q/dense/kernelsave/RestoreV2:25*
_output_shapes
:	w?*
use_locking(*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
T0
?
save/Assign_26Assignmain/q/dense/kernel/Adamsave/RestoreV2:26*
T0*
validate_shape(*
_output_shapes
:	w?*
use_locking(*&
_class
loc:@main/q/dense/kernel
?
save/Assign_27Assignmain/q/dense/kernel/Adam_1save/RestoreV2:27*&
_class
loc:@main/q/dense/kernel*
T0*
_output_shapes
:	w?*
use_locking(*
validate_shape(
?
save/Assign_28Assignmain/q/dense_1/biassave/RestoreV2:28*
T0*
validate_shape(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
use_locking(
?
save/Assign_29Assignmain/q/dense_1/bias/Adamsave/RestoreV2:29*
use_locking(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
T0
?
save/Assign_30Assignmain/q/dense_1/bias/Adam_1save/RestoreV2:30*
_output_shapes	
:?*
validate_shape(*
T0*&
_class
loc:@main/q/dense_1/bias*
use_locking(
?
save/Assign_31Assignmain/q/dense_1/kernelsave/RestoreV2:31*
validate_shape(*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
??
?
save/Assign_32Assignmain/q/dense_1/kernel/Adamsave/RestoreV2:32*(
_class
loc:@main/q/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
??*
T0
?
save/Assign_33Assignmain/q/dense_1/kernel/Adam_1save/RestoreV2:33*(
_class
loc:@main/q/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
??*
validate_shape(
?
save/Assign_34Assignmain/q/dense_2/biassave/RestoreV2:34*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*&
_class
loc:@main/q/dense_2/bias
?
save/Assign_35Assignmain/q/dense_2/bias/Adamsave/RestoreV2:35*
validate_shape(*
_output_shapes
:*
T0*&
_class
loc:@main/q/dense_2/bias*
use_locking(
?
save/Assign_36Assignmain/q/dense_2/bias/Adam_1save/RestoreV2:36*
T0*
validate_shape(*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
_output_shapes
:
?
save/Assign_37Assignmain/q/dense_2/kernelsave/RestoreV2:37*
T0*
use_locking(*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(
?
save/Assign_38Assignmain/q/dense_2/kernel/Adamsave/RestoreV2:38*
validate_shape(*
_output_shapes
:	?*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel
?
save/Assign_39Assignmain/q/dense_2/kernel/Adam_1save/RestoreV2:39*
use_locking(*
validate_shape(*
_output_shapes
:	?*
T0*(
_class
loc:@main/q/dense_2/kernel
?
save/Assign_40Assigntarget/pi/dense/biassave/RestoreV2:40*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(*'
_class
loc:@target/pi/dense/bias
?
save/Assign_41Assigntarget/pi/dense/kernelsave/RestoreV2:41*
T0*
validate_shape(*)
_class
loc:@target/pi/dense/kernel*
use_locking(*
_output_shapes
:	o?
?
save/Assign_42Assigntarget/pi/dense_1/biassave/RestoreV2:42*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(
?
save/Assign_43Assigntarget/pi/dense_1/kernelsave/RestoreV2:43*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
??
?
save/Assign_44Assigntarget/pi/dense_2/biassave/RestoreV2:44*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*)
_class
loc:@target/pi/dense_2/bias
?
save/Assign_45Assigntarget/pi/dense_2/kernelsave/RestoreV2:45*
_output_shapes
:	?*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
validate_shape(
?
save/Assign_46Assigntarget/q/dense/biassave/RestoreV2:46*
_output_shapes	
:?*&
_class
loc:@target/q/dense/bias*
T0*
use_locking(*
validate_shape(
?
save/Assign_47Assigntarget/q/dense/kernelsave/RestoreV2:47*
validate_shape(*(
_class
loc:@target/q/dense/kernel*
use_locking(*
T0*
_output_shapes
:	w?
?
save/Assign_48Assigntarget/q/dense_1/biassave/RestoreV2:48*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias
?
save/Assign_49Assigntarget/q/dense_1/kernelsave/RestoreV2:49*
use_locking(*
T0**
_class 
loc:@target/q/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(
?
save/Assign_50Assigntarget/q/dense_2/biassave/RestoreV2:50*(
_class
loc:@target/q/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
?
save/Assign_51Assigntarget/q/dense_2/kernelsave/RestoreV2:51*
T0**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	?
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
_output_shapes
: *
dtype0
?
save_1/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_e42ef09c9d534778ae888b9e099a8f4f/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
?

save_1/SaveV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
?
save_1/SaveV2/shape_and_slicesConst*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
T0*

axis *
_output_shapes
:
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
?

save_1/RestoreV2/tensor_namesConst*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0
?
!save_1/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
use_locking(*
T0*
validate_shape(
?
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
_output_shapes
: *
T0*$
_class
loc:@main/q/dense/bias*
use_locking(*
validate_shape(
?
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes
: *
use_locking(
?
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
use_locking(*$
_class
loc:@main/q/dense/bias*
_output_shapes
: *
T0*
validate_shape(
?
save_1/Assign_4Assignmain/pi/dense/biassave_1/RestoreV2:4*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?*
T0*
use_locking(
?
save_1/Assign_5Assignmain/pi/dense/bias/Adamsave_1/RestoreV2:5*
T0*%
_class
loc:@main/pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_6Assignmain/pi/dense/bias/Adam_1save_1/RestoreV2:6*
T0*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
use_locking(*
validate_shape(
?
save_1/Assign_7Assignmain/pi/dense/kernelsave_1/RestoreV2:7*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
T0*
_output_shapes
:	o?*
validate_shape(
?
save_1/Assign_8Assignmain/pi/dense/kernel/Adamsave_1/RestoreV2:8*
_output_shapes
:	o?*
T0*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
validate_shape(
?
save_1/Assign_9Assignmain/pi/dense/kernel/Adam_1save_1/RestoreV2:9*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
validate_shape(*
use_locking(*
T0
?
save_1/Assign_10Assignmain/pi/dense_1/biassave_1/RestoreV2:10*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_11Assignmain/pi/dense_1/bias/Adamsave_1/RestoreV2:11*
T0*
use_locking(*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(
?
save_1/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_1/RestoreV2:12*
validate_shape(*
_output_shapes	
:?*
T0*'
_class
loc:@main/pi/dense_1/bias*
use_locking(
?
save_1/Assign_13Assignmain/pi/dense_1/kernelsave_1/RestoreV2:13*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
??
?
save_1/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_1/RestoreV2:14*
T0*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(
?
save_1/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_1/RestoreV2:15*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
T0
?
save_1/Assign_16Assignmain/pi/dense_2/biassave_1/RestoreV2:16*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias
?
save_1/Assign_17Assignmain/pi/dense_2/bias/Adamsave_1/RestoreV2:17*
_output_shapes
:*
use_locking(*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
T0
?
save_1/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_1/RestoreV2:18*
T0*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
?
save_1/Assign_19Assignmain/pi/dense_2/kernelsave_1/RestoreV2:19*
validate_shape(*
T0*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?
?
save_1/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_1/RestoreV2:20*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	?
?
save_1/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_1/RestoreV2:21*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	?*
T0*
use_locking(
?
save_1/Assign_22Assignmain/q/dense/biassave_1/RestoreV2:22*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0*$
_class
loc:@main/q/dense/bias
?
save_1/Assign_23Assignmain/q/dense/bias/Adamsave_1/RestoreV2:23*
T0*
validate_shape(*
use_locking(*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?
?
save_1/Assign_24Assignmain/q/dense/bias/Adam_1save_1/RestoreV2:24*
T0*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
validate_shape(*
use_locking(
?
save_1/Assign_25Assignmain/q/dense/kernelsave_1/RestoreV2:25*
use_locking(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
T0
?
save_1/Assign_26Assignmain/q/dense/kernel/Adamsave_1/RestoreV2:26*
T0*
_output_shapes
:	w?*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
use_locking(
?
save_1/Assign_27Assignmain/q/dense/kernel/Adam_1save_1/RestoreV2:27*
T0*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?
?
save_1/Assign_28Assignmain/q/dense_1/biassave_1/RestoreV2:28*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_29Assignmain/q/dense_1/bias/Adamsave_1/RestoreV2:29*
use_locking(*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:?
?
save_1/Assign_30Assignmain/q/dense_1/bias/Adam_1save_1/RestoreV2:30*&
_class
loc:@main/q/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?
?
save_1/Assign_31Assignmain/q/dense_1/kernelsave_1/RestoreV2:31*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
??*
T0*
use_locking(
?
save_1/Assign_32Assignmain/q/dense_1/kernel/Adamsave_1/RestoreV2:32*
T0* 
_output_shapes
:
??*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_1/kernel
?
save_1/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_1/RestoreV2:33* 
_output_shapes
:
??*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(
?
save_1/Assign_34Assignmain/q/dense_2/biassave_1/RestoreV2:34*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
?
save_1/Assign_35Assignmain/q/dense_2/bias/Adamsave_1/RestoreV2:35*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
?
save_1/Assign_36Assignmain/q/dense_2/bias/Adam_1save_1/RestoreV2:36*
T0*
validate_shape(*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
use_locking(
?
save_1/Assign_37Assignmain/q/dense_2/kernelsave_1/RestoreV2:37*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(*(
_class
loc:@main/q/dense_2/kernel
?
save_1/Assign_38Assignmain/q/dense_2/kernel/Adamsave_1/RestoreV2:38*
T0*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	?*
validate_shape(
?
save_1/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_1/RestoreV2:39*
T0*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(
?
save_1/Assign_40Assigntarget/pi/dense/biassave_1/RestoreV2:40*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(*'
_class
loc:@target/pi/dense/bias
?
save_1/Assign_41Assigntarget/pi/dense/kernelsave_1/RestoreV2:41*
T0*
validate_shape(*
use_locking(*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?
?
save_1/Assign_42Assigntarget/pi/dense_1/biassave_1/RestoreV2:42*
validate_shape(*
_output_shapes	
:?*
T0*)
_class
loc:@target/pi/dense_1/bias*
use_locking(
?
save_1/Assign_43Assigntarget/pi/dense_1/kernelsave_1/RestoreV2:43* 
_output_shapes
:
??*
validate_shape(*
use_locking(*+
_class!
loc:@target/pi/dense_1/kernel*
T0
?
save_1/Assign_44Assigntarget/pi/dense_2/biassave_1/RestoreV2:44*
use_locking(*)
_class
loc:@target/pi/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
?
save_1/Assign_45Assigntarget/pi/dense_2/kernelsave_1/RestoreV2:45*
use_locking(*
T0*
_output_shapes
:	?*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(
?
save_1/Assign_46Assigntarget/q/dense/biassave_1/RestoreV2:46*
use_locking(*&
_class
loc:@target/q/dense/bias*
validate_shape(*
_output_shapes	
:?*
T0
?
save_1/Assign_47Assigntarget/q/dense/kernelsave_1/RestoreV2:47*
T0*(
_class
loc:@target/q/dense/kernel*
validate_shape(*
_output_shapes
:	w?*
use_locking(
?
save_1/Assign_48Assigntarget/q/dense_1/biassave_1/RestoreV2:48*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias
?
save_1/Assign_49Assigntarget/q/dense_1/kernelsave_1/RestoreV2:49*
use_locking(* 
_output_shapes
:
??*
validate_shape(**
_class 
loc:@target/q/dense_1/kernel*
T0
?
save_1/Assign_50Assigntarget/q/dense_2/biassave_1/RestoreV2:50*
validate_shape(*(
_class
loc:@target/q/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
?
save_1/Assign_51Assigntarget/q/dense_2/kernelsave_1/RestoreV2:51*
_output_shapes
:	?*
use_locking(*
validate_shape(*
T0**
_class 
loc:@target/q/dense_2/kernel
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
_output_shapes
: *
dtype0
?
save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_86752410feb4421ebaaeb82ba3ad2dd6/part*
_output_shapes
: *
dtype0
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_2/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
?
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
?

save_2/SaveV2/tensor_namesConst*
_output_shapes
:4*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
save_2/SaveV2/shape_and_slicesConst*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4
?
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_2/ShardedFilename
?
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*

axis *
_output_shapes
:*
T0*
N
?
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
?
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
_output_shapes
: *
T0
?

save_2/RestoreV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4*
dtype0
?
!save_2/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
?
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
validate_shape(*
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes
: 
?
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*$
_class
loc:@main/q/dense/bias
?
save_2/Assign_2Assignbeta2_powersave_2/RestoreV2:2*
validate_shape(*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
use_locking(
?
save_2/Assign_3Assignbeta2_power_1save_2/RestoreV2:3*
_output_shapes
: *
validate_shape(*$
_class
loc:@main/q/dense/bias*
T0*
use_locking(
?
save_2/Assign_4Assignmain/pi/dense/biassave_2/RestoreV2:4*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(*%
_class
loc:@main/pi/dense/bias
?
save_2/Assign_5Assignmain/pi/dense/bias/Adamsave_2/RestoreV2:5*
_output_shapes	
:?*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
use_locking(*
T0
?
save_2/Assign_6Assignmain/pi/dense/bias/Adam_1save_2/RestoreV2:6*
use_locking(*
T0*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
validate_shape(
?
save_2/Assign_7Assignmain/pi/dense/kernelsave_2/RestoreV2:7*
_output_shapes
:	o?*
T0*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense/kernel
?
save_2/Assign_8Assignmain/pi/dense/kernel/Adamsave_2/RestoreV2:8*
T0*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?
?
save_2/Assign_9Assignmain/pi/dense/kernel/Adam_1save_2/RestoreV2:9*
use_locking(*
validate_shape(*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?
?
save_2/Assign_10Assignmain/pi/dense_1/biassave_2/RestoreV2:10*
validate_shape(*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
T0
?
save_2/Assign_11Assignmain/pi/dense_1/bias/Adamsave_2/RestoreV2:11*
_output_shapes	
:?*
T0*
use_locking(*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(
?
save_2/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_2/RestoreV2:12*
validate_shape(*
use_locking(*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:?
?
save_2/Assign_13Assignmain/pi/dense_1/kernelsave_2/RestoreV2:13*
validate_shape(* 
_output_shapes
:
??*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel*
T0
?
save_2/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_2/RestoreV2:14* 
_output_shapes
:
??*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
T0
?
save_2/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_2/RestoreV2:15*)
_class
loc:@main/pi/dense_1/kernel*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:
??
?
save_2/Assign_16Assignmain/pi/dense_2/biassave_2/RestoreV2:16*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
?
save_2/Assign_17Assignmain/pi/dense_2/bias/Adamsave_2/RestoreV2:17*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
?
save_2/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_2/RestoreV2:18*
validate_shape(*
T0*
use_locking(*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:
?
save_2/Assign_19Assignmain/pi/dense_2/kernelsave_2/RestoreV2:19*)
_class
loc:@main/pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	?
?
save_2/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_2/RestoreV2:20*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	?*
T0
?
save_2/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_2/RestoreV2:21*
use_locking(*
validate_shape(*
_output_shapes
:	?*
T0*)
_class
loc:@main/pi/dense_2/kernel
?
save_2/Assign_22Assignmain/q/dense/biassave_2/RestoreV2:22*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias
?
save_2/Assign_23Assignmain/q/dense/bias/Adamsave_2/RestoreV2:23*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
save_2/Assign_24Assignmain/q/dense/bias/Adam_1save_2/RestoreV2:24*
T0*$
_class
loc:@main/q/dense/bias*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save_2/Assign_25Assignmain/q/dense/kernelsave_2/RestoreV2:25*
use_locking(*
_output_shapes
:	w?*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
T0
?
save_2/Assign_26Assignmain/q/dense/kernel/Adamsave_2/RestoreV2:26*
validate_shape(*
_output_shapes
:	w?*
T0*&
_class
loc:@main/q/dense/kernel*
use_locking(
?
save_2/Assign_27Assignmain/q/dense/kernel/Adam_1save_2/RestoreV2:27*&
_class
loc:@main/q/dense/kernel*
use_locking(*
_output_shapes
:	w?*
T0*
validate_shape(
?
save_2/Assign_28Assignmain/q/dense_1/biassave_2/RestoreV2:28*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
validate_shape(
?
save_2/Assign_29Assignmain/q/dense_1/bias/Adamsave_2/RestoreV2:29*
validate_shape(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
use_locking(*
T0
?
save_2/Assign_30Assignmain/q/dense_1/bias/Adam_1save_2/RestoreV2:30*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0
?
save_2/Assign_31Assignmain/q/dense_1/kernelsave_2/RestoreV2:31*
T0*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
use_locking(
?
save_2/Assign_32Assignmain/q/dense_1/kernel/Adamsave_2/RestoreV2:32*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel
?
save_2/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_2/RestoreV2:33*
T0*
use_locking(* 
_output_shapes
:
??*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel
?
save_2/Assign_34Assignmain/q/dense_2/biassave_2/RestoreV2:34*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias
?
save_2/Assign_35Assignmain/q/dense_2/bias/Adamsave_2/RestoreV2:35*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
?
save_2/Assign_36Assignmain/q/dense_2/bias/Adam_1save_2/RestoreV2:36*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
T0
?
save_2/Assign_37Assignmain/q/dense_2/kernelsave_2/RestoreV2:37*(
_class
loc:@main/q/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	?
?
save_2/Assign_38Assignmain/q/dense_2/kernel/Adamsave_2/RestoreV2:38*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel*
T0*
_output_shapes
:	?*
use_locking(
?
save_2/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_2/RestoreV2:39*
T0*
_output_shapes
:	?*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(
?
save_2/Assign_40Assigntarget/pi/dense/biassave_2/RestoreV2:40*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias
?
save_2/Assign_41Assigntarget/pi/dense/kernelsave_2/RestoreV2:41*
validate_shape(*
T0*
use_locking(*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?
?
save_2/Assign_42Assigntarget/pi/dense_1/biassave_2/RestoreV2:42*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0*)
_class
loc:@target/pi/dense_1/bias
?
save_2/Assign_43Assigntarget/pi/dense_1/kernelsave_2/RestoreV2:43*
use_locking(* 
_output_shapes
:
??*+
_class!
loc:@target/pi/dense_1/kernel*
T0*
validate_shape(
?
save_2/Assign_44Assigntarget/pi/dense_2/biassave_2/RestoreV2:44*
validate_shape(*
T0*
_output_shapes
:*)
_class
loc:@target/pi/dense_2/bias*
use_locking(
?
save_2/Assign_45Assigntarget/pi/dense_2/kernelsave_2/RestoreV2:45*+
_class!
loc:@target/pi/dense_2/kernel*
use_locking(*
_output_shapes
:	?*
T0*
validate_shape(
?
save_2/Assign_46Assigntarget/q/dense/biassave_2/RestoreV2:46*
_output_shapes	
:?*
use_locking(*&
_class
loc:@target/q/dense/bias*
validate_shape(*
T0
?
save_2/Assign_47Assigntarget/q/dense/kernelsave_2/RestoreV2:47*
T0*(
_class
loc:@target/q/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	w?
?
save_2/Assign_48Assigntarget/q/dense_1/biassave_2/RestoreV2:48*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias*
use_locking(*
validate_shape(*
T0
?
save_2/Assign_49Assigntarget/q/dense_1/kernelsave_2/RestoreV2:49**
_class 
loc:@target/q/dense_1/kernel*
T0* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save_2/Assign_50Assigntarget/q/dense_2/biassave_2/RestoreV2:50*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*(
_class
loc:@target/q/dense_2/bias
?
save_2/Assign_51Assigntarget/q/dense_2/kernelsave_2/RestoreV2:51*
validate_shape(**
_class 
loc:@target/q/dense_2/kernel*
use_locking(*
_output_shapes
:	?*
T0
?
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_5^save_2/Assign_50^save_2/Assign_51^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
shape: *
_output_shapes
: *
dtype0
?
save_3/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_5b449f190aba4c0f9a3ccb6d8fe8cc0f/part
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_3/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_3/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
?
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
?

save_3/SaveV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
?
save_3/SaveV2/shape_and_slicesConst*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
?
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
N*

axis *
_output_shapes
:*
T0
?
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
?
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
_output_shapes
: *
T0
?

save_3/RestoreV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
!save_3/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_3/AssignAssignbeta1_powersave_3/RestoreV2*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
?
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*$
_class
loc:@main/q/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
?
save_3/Assign_2Assignbeta2_powersave_3/RestoreV2:2*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
T0*
validate_shape(*
use_locking(
?
save_3/Assign_3Assignbeta2_power_1save_3/RestoreV2:3*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
?
save_3/Assign_4Assignmain/pi/dense/biassave_3/RestoreV2:4*
use_locking(*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
T0*
validate_shape(
?
save_3/Assign_5Assignmain/pi/dense/bias/Adamsave_3/RestoreV2:5*
use_locking(*
validate_shape(*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?
?
save_3/Assign_6Assignmain/pi/dense/bias/Adam_1save_3/RestoreV2:6*%
_class
loc:@main/pi/dense/bias*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
save_3/Assign_7Assignmain/pi/dense/kernelsave_3/RestoreV2:7*
use_locking(*
validate_shape(*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0
?
save_3/Assign_8Assignmain/pi/dense/kernel/Adamsave_3/RestoreV2:8*
_output_shapes
:	o?*
T0*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
validate_shape(
?
save_3/Assign_9Assignmain/pi/dense/kernel/Adam_1save_3/RestoreV2:9*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0*
validate_shape(*
use_locking(
?
save_3/Assign_10Assignmain/pi/dense_1/biassave_3/RestoreV2:10*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
save_3/Assign_11Assignmain/pi/dense_1/bias/Adamsave_3/RestoreV2:11*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
save_3/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_3/RestoreV2:12*
use_locking(*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?*
validate_shape(*
T0
?
save_3/Assign_13Assignmain/pi/dense_1/kernelsave_3/RestoreV2:13*
validate_shape(*
use_locking(* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
T0
?
save_3/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_3/RestoreV2:14*
validate_shape(*
T0* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(
?
save_3/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_3/RestoreV2:15*)
_class
loc:@main/pi/dense_1/kernel*
T0* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save_3/Assign_16Assignmain/pi/dense_2/biassave_3/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias
?
save_3/Assign_17Assignmain/pi/dense_2/bias/Adamsave_3/RestoreV2:17*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
?
save_3/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_3/RestoreV2:18*
T0*
_output_shapes
:*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
use_locking(
?
save_3/Assign_19Assignmain/pi/dense_2/kernelsave_3/RestoreV2:19*
use_locking(*
validate_shape(*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel*
T0
?
save_3/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_3/RestoreV2:20*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
use_locking(*
T0
?
save_3/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_3/RestoreV2:21*
validate_shape(*
T0*
_output_shapes
:	?*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel
?
save_3/Assign_22Assignmain/q/dense/biassave_3/RestoreV2:22*
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias*
use_locking(*
_output_shapes	
:?
?
save_3/Assign_23Assignmain/q/dense/bias/Adamsave_3/RestoreV2:23*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias
?
save_3/Assign_24Assignmain/q/dense/bias/Adam_1save_3/RestoreV2:24*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
validate_shape(*
T0*
use_locking(
?
save_3/Assign_25Assignmain/q/dense/kernelsave_3/RestoreV2:25*
T0*
use_locking(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
validate_shape(
?
save_3/Assign_26Assignmain/q/dense/kernel/Adamsave_3/RestoreV2:26*
validate_shape(*
T0*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
use_locking(
?
save_3/Assign_27Assignmain/q/dense/kernel/Adam_1save_3/RestoreV2:27*
use_locking(*
validate_shape(*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?
?
save_3/Assign_28Assignmain/q/dense_1/biassave_3/RestoreV2:28*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0*
use_locking(*
validate_shape(
?
save_3/Assign_29Assignmain/q/dense_1/bias/Adamsave_3/RestoreV2:29*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0
?
save_3/Assign_30Assignmain/q/dense_1/bias/Adam_1save_3/RestoreV2:30*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0*&
_class
loc:@main/q/dense_1/bias
?
save_3/Assign_31Assignmain/q/dense_1/kernelsave_3/RestoreV2:31* 
_output_shapes
:
??*
use_locking(*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(*
T0
?
save_3/Assign_32Assignmain/q/dense_1/kernel/Adamsave_3/RestoreV2:32*
validate_shape(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
T0*
use_locking(
?
save_3/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_3/RestoreV2:33*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
??
?
save_3/Assign_34Assignmain/q/dense_2/biassave_3/RestoreV2:34*
use_locking(*
T0*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias*
validate_shape(
?
save_3/Assign_35Assignmain/q/dense_2/bias/Adamsave_3/RestoreV2:35*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense_2/bias
?
save_3/Assign_36Assignmain/q/dense_2/bias/Adam_1save_3/RestoreV2:36*
validate_shape(*
use_locking(*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias*
T0
?
save_3/Assign_37Assignmain/q/dense_2/kernelsave_3/RestoreV2:37*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
T0*
validate_shape(*
use_locking(
?
save_3/Assign_38Assignmain/q/dense_2/kernel/Adamsave_3/RestoreV2:38*
T0*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel*
use_locking(*
_output_shapes
:	?
?
save_3/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_3/RestoreV2:39*
T0*
_output_shapes
:	?*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_2/kernel
?
save_3/Assign_40Assigntarget/pi/dense/biassave_3/RestoreV2:40*
_output_shapes	
:?*
T0*
validate_shape(*'
_class
loc:@target/pi/dense/bias*
use_locking(
?
save_3/Assign_41Assigntarget/pi/dense/kernelsave_3/RestoreV2:41*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?*
use_locking(*
T0*
validate_shape(
?
save_3/Assign_42Assigntarget/pi/dense_1/biassave_3/RestoreV2:42*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(*)
_class
loc:@target/pi/dense_1/bias
?
save_3/Assign_43Assigntarget/pi/dense_1/kernelsave_3/RestoreV2:43*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(*
T0
?
save_3/Assign_44Assigntarget/pi/dense_2/biassave_3/RestoreV2:44*
T0*)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save_3/Assign_45Assigntarget/pi/dense_2/kernelsave_3/RestoreV2:45*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	?*
validate_shape(*
T0
?
save_3/Assign_46Assigntarget/q/dense/biassave_3/RestoreV2:46*
T0*
use_locking(*&
_class
loc:@target/q/dense/bias*
_output_shapes	
:?*
validate_shape(
?
save_3/Assign_47Assigntarget/q/dense/kernelsave_3/RestoreV2:47*
_output_shapes
:	w?*
validate_shape(*
T0*
use_locking(*(
_class
loc:@target/q/dense/kernel
?
save_3/Assign_48Assigntarget/q/dense_1/biassave_3/RestoreV2:48*(
_class
loc:@target/q/dense_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
save_3/Assign_49Assigntarget/q/dense_1/kernelsave_3/RestoreV2:49**
_class 
loc:@target/q/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
??*
T0
?
save_3/Assign_50Assigntarget/q/dense_2/biassave_3/RestoreV2:50*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*(
_class
loc:@target/q/dense_2/bias
?
save_3/Assign_51Assigntarget/q/dense_2/kernelsave_3/RestoreV2:51*
T0*
validate_shape(**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:	?*
use_locking(
?
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_51^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
dtype0*
shape: *
_output_shapes
: 
?
save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_16a494005ff3462286521e48e08da873/part*
dtype0*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_4/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
?
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
?

save_4/SaveV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
?
save_4/SaveV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
?
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_4/ShardedFilename
?
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
N*

axis *
T0*
_output_shapes
:
?
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
?
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
_output_shapes
: *
T0
?

save_4/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
!save_4/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
?
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*B
dtypes8
624*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
_output_shapes
: *
use_locking(*
T0*
validate_shape(*%
_class
loc:@main/pi/dense/bias
?
save_4/Assign_1Assignbeta1_power_1save_4/RestoreV2:1*$
_class
loc:@main/q/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
?
save_4/Assign_2Assignbeta2_powersave_4/RestoreV2:2*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
?
save_4/Assign_3Assignbeta2_power_1save_4/RestoreV2:3*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
?
save_4/Assign_4Assignmain/pi/dense/biassave_4/RestoreV2:4*
validate_shape(*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
use_locking(*
T0
?
save_4/Assign_5Assignmain/pi/dense/bias/Adamsave_4/RestoreV2:5*%
_class
loc:@main/pi/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
?
save_4/Assign_6Assignmain/pi/dense/bias/Adam_1save_4/RestoreV2:6*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
T0*
use_locking(
?
save_4/Assign_7Assignmain/pi/dense/kernelsave_4/RestoreV2:7*
T0*
_output_shapes
:	o?*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
validate_shape(
?
save_4/Assign_8Assignmain/pi/dense/kernel/Adamsave_4/RestoreV2:8*
validate_shape(*
_output_shapes
:	o?*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
T0
?
save_4/Assign_9Assignmain/pi/dense/kernel/Adam_1save_4/RestoreV2:9*
T0*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
validate_shape(
?
save_4/Assign_10Assignmain/pi/dense_1/biassave_4/RestoreV2:10*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
save_4/Assign_11Assignmain/pi/dense_1/bias/Adamsave_4/RestoreV2:11*
_output_shapes	
:?*
T0*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
validate_shape(
?
save_4/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_4/RestoreV2:12*
T0*
validate_shape(*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
use_locking(
?
save_4/Assign_13Assignmain/pi/dense_1/kernelsave_4/RestoreV2:13*
validate_shape(*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
T0
?
save_4/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_4/RestoreV2:14*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(*
T0
?
save_4/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_4/RestoreV2:15* 
_output_shapes
:
??*
use_locking(*
T0*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel
?
save_4/Assign_16Assignmain/pi/dense_2/biassave_4/RestoreV2:16*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(
?
save_4/Assign_17Assignmain/pi/dense_2/bias/Adamsave_4/RestoreV2:17*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*'
_class
loc:@main/pi/dense_2/bias
?
save_4/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_4/RestoreV2:18*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
?
save_4/Assign_19Assignmain/pi/dense_2/kernelsave_4/RestoreV2:19*
validate_shape(*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
T0
?
save_4/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_4/RestoreV2:20*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel
?
save_4/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_4/RestoreV2:21*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel
?
save_4/Assign_22Assignmain/q/dense/biassave_4/RestoreV2:22*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_4/Assign_23Assignmain/q/dense/bias/Adamsave_4/RestoreV2:23*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias
?
save_4/Assign_24Assignmain/q/dense/bias/Adam_1save_4/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
T0
?
save_4/Assign_25Assignmain/q/dense/kernelsave_4/RestoreV2:25*
validate_shape(*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?
?
save_4/Assign_26Assignmain/q/dense/kernel/Adamsave_4/RestoreV2:26*
use_locking(*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	w?
?
save_4/Assign_27Assignmain/q/dense/kernel/Adam_1save_4/RestoreV2:27*
_output_shapes
:	w?*
use_locking(*&
_class
loc:@main/q/dense/kernel*
T0*
validate_shape(
?
save_4/Assign_28Assignmain/q/dense_1/biassave_4/RestoreV2:28*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0*&
_class
loc:@main/q/dense_1/bias
?
save_4/Assign_29Assignmain/q/dense_1/bias/Adamsave_4/RestoreV2:29*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(
?
save_4/Assign_30Assignmain/q/dense_1/bias/Adam_1save_4/RestoreV2:30*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(
?
save_4/Assign_31Assignmain/q/dense_1/kernelsave_4/RestoreV2:31* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
use_locking(*
validate_shape(*
T0
?
save_4/Assign_32Assignmain/q/dense_1/kernel/Adamsave_4/RestoreV2:32*
T0*
use_locking(*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
??
?
save_4/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_4/RestoreV2:33*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_1/kernel*
T0* 
_output_shapes
:
??
?
save_4/Assign_34Assignmain/q/dense_2/biassave_4/RestoreV2:34*
T0*
_output_shapes
:*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
validate_shape(
?
save_4/Assign_35Assignmain/q/dense_2/bias/Adamsave_4/RestoreV2:35*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*&
_class
loc:@main/q/dense_2/bias
?
save_4/Assign_36Assignmain/q/dense_2/bias/Adam_1save_4/RestoreV2:36*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*&
_class
loc:@main/q/dense_2/bias
?
save_4/Assign_37Assignmain/q/dense_2/kernelsave_4/RestoreV2:37*
validate_shape(*
use_locking(*
_output_shapes
:	?*
T0*(
_class
loc:@main/q/dense_2/kernel
?
save_4/Assign_38Assignmain/q/dense_2/kernel/Adamsave_4/RestoreV2:38*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
use_locking(*
T0
?
save_4/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_4/RestoreV2:39*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	?*
use_locking(
?
save_4/Assign_40Assigntarget/pi/dense/biassave_4/RestoreV2:40*
use_locking(*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
T0*
_output_shapes	
:?
?
save_4/Assign_41Assigntarget/pi/dense/kernelsave_4/RestoreV2:41*
validate_shape(*
_output_shapes
:	o?*)
_class
loc:@target/pi/dense/kernel*
T0*
use_locking(
?
save_4/Assign_42Assigntarget/pi/dense_1/biassave_4/RestoreV2:42*
T0*)
_class
loc:@target/pi/dense_1/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save_4/Assign_43Assigntarget/pi/dense_1/kernelsave_4/RestoreV2:43*
validate_shape(*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
??*
T0*
use_locking(
?
save_4/Assign_44Assigntarget/pi/dense_2/biassave_4/RestoreV2:44*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*)
_class
loc:@target/pi/dense_2/bias
?
save_4/Assign_45Assigntarget/pi/dense_2/kernelsave_4/RestoreV2:45*
validate_shape(*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
save_4/Assign_46Assigntarget/q/dense/biassave_4/RestoreV2:46*
use_locking(*&
_class
loc:@target/q/dense/bias*
validate_shape(*
T0*
_output_shapes	
:?
?
save_4/Assign_47Assigntarget/q/dense/kernelsave_4/RestoreV2:47*(
_class
loc:@target/q/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	w?
?
save_4/Assign_48Assigntarget/q/dense_1/biassave_4/RestoreV2:48*(
_class
loc:@target/q/dense_1/bias*
_output_shapes	
:?*
T0*
use_locking(*
validate_shape(
?
save_4/Assign_49Assigntarget/q/dense_1/kernelsave_4/RestoreV2:49**
_class 
loc:@target/q/dense_1/kernel*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:
??
?
save_4/Assign_50Assigntarget/q/dense_2/biassave_4/RestoreV2:50*
_output_shapes
:*
T0*(
_class
loc:@target/q/dense_2/bias*
use_locking(*
validate_shape(
?
save_4/Assign_51Assigntarget/q/dense_2/kernelsave_4/RestoreV2:51*
use_locking(**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	?*
T0
?
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_45^save_4/Assign_46^save_4/Assign_47^save_4/Assign_48^save_4/Assign_49^save_4/Assign_5^save_4/Assign_50^save_4/Assign_51^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
shape: *
_output_shapes
: *
dtype0
?
save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_3562815cd7e74a159a0fc29cc205f22f/part*
_output_shapes
: *
dtype0
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_5/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
?
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
?

save_5/SaveV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4*
dtype0
?
save_5/SaveV2/shape_and_slicesConst*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4
?
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: *
T0
?
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
_output_shapes
:*

axis *
T0*
N
?
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
?
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
_output_shapes
: *
T0
?

save_5/RestoreV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
!save_5/RestoreV2/shape_and_slicesConst*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*B
dtypes8
624*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias
?
save_5/Assign_1Assignbeta1_power_1save_5/RestoreV2:1*
use_locking(*
_output_shapes
: *
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias
?
save_5/Assign_2Assignbeta2_powersave_5/RestoreV2:2*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
?
save_5/Assign_3Assignbeta2_power_1save_5/RestoreV2:3*$
_class
loc:@main/q/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
?
save_5/Assign_4Assignmain/pi/dense/biassave_5/RestoreV2:4*%
_class
loc:@main/pi/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_5/Assign_5Assignmain/pi/dense/bias/Adamsave_5/RestoreV2:5*
use_locking(*
_output_shapes	
:?*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
T0
?
save_5/Assign_6Assignmain/pi/dense/bias/Adam_1save_5/RestoreV2:6*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
T0*
use_locking(*
_output_shapes	
:?
?
save_5/Assign_7Assignmain/pi/dense/kernelsave_5/RestoreV2:7*
_output_shapes
:	o?*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
T0*
use_locking(
?
save_5/Assign_8Assignmain/pi/dense/kernel/Adamsave_5/RestoreV2:8*
validate_shape(*
use_locking(*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0
?
save_5/Assign_9Assignmain/pi/dense/kernel/Adam_1save_5/RestoreV2:9*
validate_shape(*
_output_shapes
:	o?*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
T0
?
save_5/Assign_10Assignmain/pi/dense_1/biassave_5/RestoreV2:10*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:?
?
save_5/Assign_11Assignmain/pi/dense_1/bias/Adamsave_5/RestoreV2:11*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0*'
_class
loc:@main/pi/dense_1/bias
?
save_5/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_5/RestoreV2:12*'
_class
loc:@main/pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_5/Assign_13Assignmain/pi/dense_1/kernelsave_5/RestoreV2:13*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
??
?
save_5/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_5/RestoreV2:14* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
?
save_5/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_5/RestoreV2:15*
validate_shape(* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(*
T0
?
save_5/Assign_16Assignmain/pi/dense_2/biassave_5/RestoreV2:16*
validate_shape(*
use_locking(*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
T0
?
save_5/Assign_17Assignmain/pi/dense_2/bias/Adamsave_5/RestoreV2:17*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*'
_class
loc:@main/pi/dense_2/bias
?
save_5/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_5/RestoreV2:18*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias
?
save_5/Assign_19Assignmain/pi/dense_2/kernelsave_5/RestoreV2:19*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
T0*
validate_shape(
?
save_5/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_5/RestoreV2:20*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	?
?
save_5/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_5/RestoreV2:21*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
save_5/Assign_22Assignmain/q/dense/biassave_5/RestoreV2:22*$
_class
loc:@main/q/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?
?
save_5/Assign_23Assignmain/q/dense/bias/Adamsave_5/RestoreV2:23*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
?
save_5/Assign_24Assignmain/q/dense/bias/Adam_1save_5/RestoreV2:24*
validate_shape(*$
_class
loc:@main/q/dense/bias*
use_locking(*
_output_shapes	
:?*
T0
?
save_5/Assign_25Assignmain/q/dense/kernelsave_5/RestoreV2:25*
T0*
_output_shapes
:	w?*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense/kernel
?
save_5/Assign_26Assignmain/q/dense/kernel/Adamsave_5/RestoreV2:26*
T0*
_output_shapes
:	w?*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
use_locking(
?
save_5/Assign_27Assignmain/q/dense/kernel/Adam_1save_5/RestoreV2:27*
_output_shapes
:	w?*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
T0*
use_locking(
?
save_5/Assign_28Assignmain/q/dense_1/biassave_5/RestoreV2:28*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias
?
save_5/Assign_29Assignmain/q/dense_1/bias/Adamsave_5/RestoreV2:29*
T0*&
_class
loc:@main/q/dense_1/bias*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save_5/Assign_30Assignmain/q/dense_1/bias/Adam_1save_5/RestoreV2:30*
T0*
validate_shape(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
use_locking(
?
save_5/Assign_31Assignmain/q/dense_1/kernelsave_5/RestoreV2:31*
validate_shape(*
use_locking(* 
_output_shapes
:
??*
T0*(
_class
loc:@main/q/dense_1/kernel
?
save_5/Assign_32Assignmain/q/dense_1/kernel/Adamsave_5/RestoreV2:32* 
_output_shapes
:
??*
validate_shape(*
T0*
use_locking(*(
_class
loc:@main/q/dense_1/kernel
?
save_5/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_5/RestoreV2:33*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel
?
save_5/Assign_34Assignmain/q/dense_2/biassave_5/RestoreV2:34*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
T0*
validate_shape(
?
save_5/Assign_35Assignmain/q/dense_2/bias/Adamsave_5/RestoreV2:35*
validate_shape(*
T0*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
_output_shapes
:
?
save_5/Assign_36Assignmain/q/dense_2/bias/Adam_1save_5/RestoreV2:36*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
?
save_5/Assign_37Assignmain/q/dense_2/kernelsave_5/RestoreV2:37*
validate_shape(*
_output_shapes
:	?*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel
?
save_5/Assign_38Assignmain/q/dense_2/kernel/Adamsave_5/RestoreV2:38*
validate_shape(*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
T0*
use_locking(
?
save_5/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_5/RestoreV2:39*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(*(
_class
loc:@main/q/dense_2/kernel
?
save_5/Assign_40Assigntarget/pi/dense/biassave_5/RestoreV2:40*'
_class
loc:@target/pi/dense/bias*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(
?
save_5/Assign_41Assigntarget/pi/dense/kernelsave_5/RestoreV2:41*
_output_shapes
:	o?*
T0*
validate_shape(*
use_locking(*)
_class
loc:@target/pi/dense/kernel
?
save_5/Assign_42Assigntarget/pi/dense_1/biassave_5/RestoreV2:42*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(*)
_class
loc:@target/pi/dense_1/bias
?
save_5/Assign_43Assigntarget/pi/dense_1/kernelsave_5/RestoreV2:43*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
??*+
_class!
loc:@target/pi/dense_1/kernel
?
save_5/Assign_44Assigntarget/pi/dense_2/biassave_5/RestoreV2:44*
validate_shape(*)
_class
loc:@target/pi/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
?
save_5/Assign_45Assigntarget/pi/dense_2/kernelsave_5/RestoreV2:45*
T0*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	?
?
save_5/Assign_46Assigntarget/q/dense/biassave_5/RestoreV2:46*
validate_shape(*
_output_shapes	
:?*
use_locking(*&
_class
loc:@target/q/dense/bias*
T0
?
save_5/Assign_47Assigntarget/q/dense/kernelsave_5/RestoreV2:47*(
_class
loc:@target/q/dense/kernel*
validate_shape(*
_output_shapes
:	w?*
T0*
use_locking(
?
save_5/Assign_48Assigntarget/q/dense_1/biassave_5/RestoreV2:48*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_5/Assign_49Assigntarget/q/dense_1/kernelsave_5/RestoreV2:49*
use_locking(* 
_output_shapes
:
??*
T0*
validate_shape(**
_class 
loc:@target/q/dense_1/kernel
?
save_5/Assign_50Assigntarget/q/dense_2/biassave_5/RestoreV2:50*
validate_shape(*
T0*
_output_shapes
:*(
_class
loc:@target/q/dense_2/bias*
use_locking(
?
save_5/Assign_51Assigntarget/q/dense_2/kernelsave_5/RestoreV2:51**
_class 
loc:@target/q/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	?*
validate_shape(
?
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_45^save_5/Assign_46^save_5/Assign_47^save_5/Assign_48^save_5/Assign_49^save_5/Assign_5^save_5/Assign_50^save_5/Assign_51^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
shape: *
_output_shapes
: *
dtype0
?
save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_4ac36ac26af4427aa37ade0a98087a7c/part*
dtype0*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_6/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
?
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
?

save_6/SaveV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
save_6/SaveV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
?
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: *
T0
?
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
N*
T0*

axis *
_output_shapes
:
?
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
?
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
_output_shapes
: *
T0
?

save_6/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
!save_6/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
T0*
use_locking(*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
?
save_6/Assign_1Assignbeta1_power_1save_6/RestoreV2:1*
use_locking(*
_output_shapes
: *
validate_shape(*
T0*$
_class
loc:@main/q/dense/bias
?
save_6/Assign_2Assignbeta2_powersave_6/RestoreV2:2*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
?
save_6/Assign_3Assignbeta2_power_1save_6/RestoreV2:3*
_output_shapes
: *$
_class
loc:@main/q/dense/bias*
use_locking(*
validate_shape(*
T0
?
save_6/Assign_4Assignmain/pi/dense/biassave_6/RestoreV2:4*
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
save_6/Assign_5Assignmain/pi/dense/bias/Adamsave_6/RestoreV2:5*
use_locking(*
T0*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
validate_shape(
?
save_6/Assign_6Assignmain/pi/dense/bias/Adam_1save_6/RestoreV2:6*
validate_shape(*
_output_shapes	
:?*
T0*%
_class
loc:@main/pi/dense/bias*
use_locking(
?
save_6/Assign_7Assignmain/pi/dense/kernelsave_6/RestoreV2:7*
validate_shape(*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0*
use_locking(
?
save_6/Assign_8Assignmain/pi/dense/kernel/Adamsave_6/RestoreV2:8*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
use_locking(*
T0
?
save_6/Assign_9Assignmain/pi/dense/kernel/Adam_1save_6/RestoreV2:9*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
validate_shape(*
T0
?
save_6/Assign_10Assignmain/pi/dense_1/biassave_6/RestoreV2:10*
use_locking(*
T0*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?
?
save_6/Assign_11Assignmain/pi/dense_1/bias/Adamsave_6/RestoreV2:11*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?
?
save_6/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_6/RestoreV2:12*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
use_locking(*
T0
?
save_6/Assign_13Assignmain/pi/dense_1/kernelsave_6/RestoreV2:13*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
??
?
save_6/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_6/RestoreV2:14*
validate_shape(*
T0*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??
?
save_6/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_6/RestoreV2:15*
use_locking(*
T0* 
_output_shapes
:
??*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel
?
save_6/Assign_16Assignmain/pi/dense_2/biassave_6/RestoreV2:16*
use_locking(*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
T0
?
save_6/Assign_17Assignmain/pi/dense_2/bias/Adamsave_6/RestoreV2:17*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
?
save_6/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_6/RestoreV2:18*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*'
_class
loc:@main/pi/dense_2/bias
?
save_6/Assign_19Assignmain/pi/dense_2/kernelsave_6/RestoreV2:19*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	?
?
save_6/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_6/RestoreV2:20*
validate_shape(*
use_locking(*
_output_shapes
:	?*
T0*)
_class
loc:@main/pi/dense_2/kernel
?
save_6/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_6/RestoreV2:21*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
T0*
use_locking(
?
save_6/Assign_22Assignmain/q/dense/biassave_6/RestoreV2:22*
use_locking(*$
_class
loc:@main/q/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
save_6/Assign_23Assignmain/q/dense/bias/Adamsave_6/RestoreV2:23*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias
?
save_6/Assign_24Assignmain/q/dense/bias/Adam_1save_6/RestoreV2:24*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
use_locking(*
T0*
validate_shape(
?
save_6/Assign_25Assignmain/q/dense/kernelsave_6/RestoreV2:25*
_output_shapes
:	w?*
T0*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense/kernel
?
save_6/Assign_26Assignmain/q/dense/kernel/Adamsave_6/RestoreV2:26*&
_class
loc:@main/q/dense/kernel*
T0*
use_locking(*
_output_shapes
:	w?*
validate_shape(
?
save_6/Assign_27Assignmain/q/dense/kernel/Adam_1save_6/RestoreV2:27*
validate_shape(*
_output_shapes
:	w?*
T0*
use_locking(*&
_class
loc:@main/q/dense/kernel
?
save_6/Assign_28Assignmain/q/dense_1/biassave_6/RestoreV2:28*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:?
?
save_6/Assign_29Assignmain/q/dense_1/bias/Adamsave_6/RestoreV2:29*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(
?
save_6/Assign_30Assignmain/q/dense_1/bias/Adam_1save_6/RestoreV2:30*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense_1/bias
?
save_6/Assign_31Assignmain/q/dense_1/kernelsave_6/RestoreV2:31*
use_locking(* 
_output_shapes
:
??*
validate_shape(*
T0*(
_class
loc:@main/q/dense_1/kernel
?
save_6/Assign_32Assignmain/q/dense_1/kernel/Adamsave_6/RestoreV2:32*
validate_shape(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
T0*
use_locking(
?
save_6/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_6/RestoreV2:33*(
_class
loc:@main/q/dense_1/kernel*
T0* 
_output_shapes
:
??*
validate_shape(*
use_locking(
?
save_6/Assign_34Assignmain/q/dense_2/biassave_6/RestoreV2:34*
T0*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save_6/Assign_35Assignmain/q/dense_2/bias/Adamsave_6/RestoreV2:35*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
?
save_6/Assign_36Assignmain/q/dense_2/bias/Adam_1save_6/RestoreV2:36*
validate_shape(*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
?
save_6/Assign_37Assignmain/q/dense_2/kernelsave_6/RestoreV2:37*
use_locking(*
_output_shapes
:	?*
T0*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel
?
save_6/Assign_38Assignmain/q/dense_2/kernel/Adamsave_6/RestoreV2:38*
_output_shapes
:	?*
T0*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_2/kernel
?
save_6/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_6/RestoreV2:39*(
_class
loc:@main/q/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	?*
validate_shape(
?
save_6/Assign_40Assigntarget/pi/dense/biassave_6/RestoreV2:40*
T0*
_output_shapes	
:?*
use_locking(*'
_class
loc:@target/pi/dense/bias*
validate_shape(
?
save_6/Assign_41Assigntarget/pi/dense/kernelsave_6/RestoreV2:41*
T0*
_output_shapes
:	o?*
use_locking(*
validate_shape(*)
_class
loc:@target/pi/dense/kernel
?
save_6/Assign_42Assigntarget/pi/dense_1/biassave_6/RestoreV2:42*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
save_6/Assign_43Assigntarget/pi/dense_1/kernelsave_6/RestoreV2:43*+
_class!
loc:@target/pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
??
?
save_6/Assign_44Assigntarget/pi/dense_2/biassave_6/RestoreV2:44*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*)
_class
loc:@target/pi/dense_2/bias
?
save_6/Assign_45Assigntarget/pi/dense_2/kernelsave_6/RestoreV2:45*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	?
?
save_6/Assign_46Assigntarget/q/dense/biassave_6/RestoreV2:46*
validate_shape(*
T0*&
_class
loc:@target/q/dense/bias*
_output_shapes	
:?*
use_locking(
?
save_6/Assign_47Assigntarget/q/dense/kernelsave_6/RestoreV2:47*
use_locking(*
_output_shapes
:	w?*
validate_shape(*
T0*(
_class
loc:@target/q/dense/kernel
?
save_6/Assign_48Assigntarget/q/dense_1/biassave_6/RestoreV2:48*
T0*
use_locking(*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias*
validate_shape(
?
save_6/Assign_49Assigntarget/q/dense_1/kernelsave_6/RestoreV2:49* 
_output_shapes
:
??**
_class 
loc:@target/q/dense_1/kernel*
validate_shape(*
use_locking(*
T0
?
save_6/Assign_50Assigntarget/q/dense_2/biassave_6/RestoreV2:50*(
_class
loc:@target/q/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
?
save_6/Assign_51Assigntarget/q/dense_2/kernelsave_6/RestoreV2:51*
T0*
use_locking(*
_output_shapes
:	?**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(
?
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_45^save_6/Assign_46^save_6/Assign_47^save_6/Assign_48^save_6/Assign_49^save_6/Assign_5^save_6/Assign_50^save_6/Assign_51^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
_output_shapes
: *
dtype0*
shape: 
?
save_7/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_7b9e7ac94bdf4052b8efdc6f5f5c9135/part
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_7/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
?
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
?

save_7/SaveV2/tensor_namesConst*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0
?
save_7/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_7/ShardedFilename
?
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*
_output_shapes
:*
N*
T0*

axis 
?
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
?
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
?

save_7/RestoreV2/tensor_namesConst*
_output_shapes
:4*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
!save_7/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
use_locking(*
_output_shapes
: *
T0
?
save_7/Assign_1Assignbeta1_power_1save_7/RestoreV2:1*
validate_shape(*
T0*$
_class
loc:@main/q/dense/bias*
_output_shapes
: *
use_locking(
?
save_7/Assign_2Assignbeta2_powersave_7/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(
?
save_7/Assign_3Assignbeta2_power_1save_7/RestoreV2:3*
_output_shapes
: *
use_locking(*$
_class
loc:@main/q/dense/bias*
validate_shape(*
T0
?
save_7/Assign_4Assignmain/pi/dense/biassave_7/RestoreV2:4*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0
?
save_7/Assign_5Assignmain/pi/dense/bias/Adamsave_7/RestoreV2:5*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(*%
_class
loc:@main/pi/dense/bias
?
save_7/Assign_6Assignmain/pi/dense/bias/Adam_1save_7/RestoreV2:6*
_output_shapes	
:?*
T0*
use_locking(*%
_class
loc:@main/pi/dense/bias*
validate_shape(
?
save_7/Assign_7Assignmain/pi/dense/kernelsave_7/RestoreV2:7*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
validate_shape(*
T0
?
save_7/Assign_8Assignmain/pi/dense/kernel/Adamsave_7/RestoreV2:8*
T0*
_output_shapes
:	o?*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
validate_shape(
?
save_7/Assign_9Assignmain/pi/dense/kernel/Adam_1save_7/RestoreV2:9*
validate_shape(*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0*
use_locking(
?
save_7/Assign_10Assignmain/pi/dense_1/biassave_7/RestoreV2:10*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias
?
save_7/Assign_11Assignmain/pi/dense_1/bias/Adamsave_7/RestoreV2:11*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(
?
save_7/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_7/RestoreV2:12*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias
?
save_7/Assign_13Assignmain/pi/dense_1/kernelsave_7/RestoreV2:13*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
??
?
save_7/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_7/RestoreV2:14*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
T0* 
_output_shapes
:
??
?
save_7/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_7/RestoreV2:15*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
use_locking(*
T0
?
save_7/Assign_16Assignmain/pi/dense_2/biassave_7/RestoreV2:16*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*'
_class
loc:@main/pi/dense_2/bias
?
save_7/Assign_17Assignmain/pi/dense_2/bias/Adamsave_7/RestoreV2:17*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*'
_class
loc:@main/pi/dense_2/bias
?
save_7/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_7/RestoreV2:18*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
T0*
validate_shape(
?
save_7/Assign_19Assignmain/pi/dense_2/kernelsave_7/RestoreV2:19*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
save_7/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_7/RestoreV2:20*
T0*
_output_shapes
:	?*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(
?
save_7/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_7/RestoreV2:21*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(*
T0
?
save_7/Assign_22Assignmain/q/dense/biassave_7/RestoreV2:22*
validate_shape(*$
_class
loc:@main/q/dense/bias*
T0*
use_locking(*
_output_shapes	
:?
?
save_7/Assign_23Assignmain/q/dense/bias/Adamsave_7/RestoreV2:23*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_7/Assign_24Assignmain/q/dense/bias/Adam_1save_7/RestoreV2:24*
use_locking(*$
_class
loc:@main/q/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
save_7/Assign_25Assignmain/q/dense/kernelsave_7/RestoreV2:25*
_output_shapes
:	w?*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense/kernel*
T0
?
save_7/Assign_26Assignmain/q/dense/kernel/Adamsave_7/RestoreV2:26*
T0*
validate_shape(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
use_locking(
?
save_7/Assign_27Assignmain/q/dense/kernel/Adam_1save_7/RestoreV2:27*&
_class
loc:@main/q/dense/kernel*
T0*
use_locking(*
_output_shapes
:	w?*
validate_shape(
?
save_7/Assign_28Assignmain/q/dense_1/biassave_7/RestoreV2:28*
use_locking(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
save_7/Assign_29Assignmain/q/dense_1/bias/Adamsave_7/RestoreV2:29*
T0*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
use_locking(
?
save_7/Assign_30Assignmain/q/dense_1/bias/Adam_1save_7/RestoreV2:30*
use_locking(*
_output_shapes	
:?*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
T0
?
save_7/Assign_31Assignmain/q/dense_1/kernelsave_7/RestoreV2:31*
T0* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(*
use_locking(
?
save_7/Assign_32Assignmain/q/dense_1/kernel/Adamsave_7/RestoreV2:32*
T0*
use_locking(*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
??
?
save_7/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_7/RestoreV2:33*(
_class
loc:@main/q/dense_1/kernel*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
??
?
save_7/Assign_34Assignmain/q/dense_2/biassave_7/RestoreV2:34*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense_2/bias
?
save_7/Assign_35Assignmain/q/dense_2/bias/Adamsave_7/RestoreV2:35*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
?
save_7/Assign_36Assignmain/q/dense_2/bias/Adam_1save_7/RestoreV2:36*
_output_shapes
:*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
T0
?
save_7/Assign_37Assignmain/q/dense_2/kernelsave_7/RestoreV2:37*
validate_shape(*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
use_locking(*
T0
?
save_7/Assign_38Assignmain/q/dense_2/kernel/Adamsave_7/RestoreV2:38*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
use_locking(*
validate_shape(*
T0
?
save_7/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_7/RestoreV2:39*
_output_shapes
:	?*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel*
T0*
use_locking(
?
save_7/Assign_40Assigntarget/pi/dense/biassave_7/RestoreV2:40*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(*'
_class
loc:@target/pi/dense/bias
?
save_7/Assign_41Assigntarget/pi/dense/kernelsave_7/RestoreV2:41*
_output_shapes
:	o?*
validate_shape(*
T0*
use_locking(*)
_class
loc:@target/pi/dense/kernel
?
save_7/Assign_42Assigntarget/pi/dense_1/biassave_7/RestoreV2:42*
T0*
_output_shapes	
:?*)
_class
loc:@target/pi/dense_1/bias*
use_locking(*
validate_shape(
?
save_7/Assign_43Assigntarget/pi/dense_1/kernelsave_7/RestoreV2:43*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
??
?
save_7/Assign_44Assigntarget/pi/dense_2/biassave_7/RestoreV2:44*
_output_shapes
:*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
use_locking(
?
save_7/Assign_45Assigntarget/pi/dense_2/kernelsave_7/RestoreV2:45*
validate_shape(*+
_class!
loc:@target/pi/dense_2/kernel*
use_locking(*
_output_shapes
:	?*
T0
?
save_7/Assign_46Assigntarget/q/dense/biassave_7/RestoreV2:46*
_output_shapes	
:?*
use_locking(*
validate_shape(*&
_class
loc:@target/q/dense/bias*
T0
?
save_7/Assign_47Assigntarget/q/dense/kernelsave_7/RestoreV2:47*(
_class
loc:@target/q/dense/kernel*
_output_shapes
:	w?*
T0*
use_locking(*
validate_shape(
?
save_7/Assign_48Assigntarget/q/dense_1/biassave_7/RestoreV2:48*(
_class
loc:@target/q/dense_1/bias*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
save_7/Assign_49Assigntarget/q/dense_1/kernelsave_7/RestoreV2:49**
_class 
loc:@target/q/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_7/Assign_50Assigntarget/q/dense_2/biassave_7/RestoreV2:50*(
_class
loc:@target/q/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
?
save_7/Assign_51Assigntarget/q/dense_2/kernelsave_7/RestoreV2:51*
T0*
use_locking(*
validate_shape(**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:	?
?
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_45^save_7/Assign_46^save_7/Assign_47^save_7/Assign_48^save_7/Assign_49^save_7/Assign_5^save_7/Assign_50^save_7/Assign_51^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
shape: *
_output_shapes
: *
dtype0
?
save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_86382633f41a44c1b67daf617a64c84b/part*
_output_shapes
: *
dtype0
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_8/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_8/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
?
save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
?

save_8/SaveV2/tensor_namesConst*
_output_shapes
:4*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
save_8/SaveV2/shape_and_slicesConst*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*)
_class
loc:@save_8/ShardedFilename*
T0*
_output_shapes
: 
?
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
N*
T0*

axis *
_output_shapes
:
?
save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
?
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
T0*
_output_shapes
: 
?

save_8/RestoreV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
?
!save_8/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
?
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_8/AssignAssignbeta1_powersave_8/RestoreV2*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
?
save_8/Assign_1Assignbeta1_power_1save_8/RestoreV2:1*
use_locking(*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes
: *
T0
?
save_8/Assign_2Assignbeta2_powersave_8/RestoreV2:2*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*%
_class
loc:@main/pi/dense/bias
?
save_8/Assign_3Assignbeta2_power_1save_8/RestoreV2:3*
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias*
_output_shapes
: *
use_locking(
?
save_8/Assign_4Assignmain/pi/dense/biassave_8/RestoreV2:4*
T0*
use_locking(*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?
?
save_8/Assign_5Assignmain/pi/dense/bias/Adamsave_8/RestoreV2:5*
validate_shape(*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
use_locking(*
T0
?
save_8/Assign_6Assignmain/pi/dense/bias/Adam_1save_8/RestoreV2:6*
_output_shapes	
:?*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(
?
save_8/Assign_7Assignmain/pi/dense/kernelsave_8/RestoreV2:7*'
_class
loc:@main/pi/dense/kernel*
T0*
use_locking(*
_output_shapes
:	o?*
validate_shape(
?
save_8/Assign_8Assignmain/pi/dense/kernel/Adamsave_8/RestoreV2:8*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	o?
?
save_8/Assign_9Assignmain/pi/dense/kernel/Adam_1save_8/RestoreV2:9*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
validate_shape(*
T0
?
save_8/Assign_10Assignmain/pi/dense_1/biassave_8/RestoreV2:10*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
save_8/Assign_11Assignmain/pi/dense_1/bias/Adamsave_8/RestoreV2:11*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(
?
save_8/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_8/RestoreV2:12*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?
?
save_8/Assign_13Assignmain/pi/dense_1/kernelsave_8/RestoreV2:13*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(* 
_output_shapes
:
??*
T0
?
save_8/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_8/RestoreV2:14* 
_output_shapes
:
??*
use_locking(*
T0*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel
?
save_8/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_8/RestoreV2:15*
validate_shape(*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??
?
save_8/Assign_16Assignmain/pi/dense_2/biassave_8/RestoreV2:16*
use_locking(*
_output_shapes
:*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(
?
save_8/Assign_17Assignmain/pi/dense_2/bias/Adamsave_8/RestoreV2:17*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(
?
save_8/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_8/RestoreV2:18*
_output_shapes
:*
use_locking(*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
T0
?
save_8/Assign_19Assignmain/pi/dense_2/kernelsave_8/RestoreV2:19*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
T0
?
save_8/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_8/RestoreV2:20*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
validate_shape(*
use_locking(*
T0
?
save_8/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_8/RestoreV2:21*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
T0*
use_locking(
?
save_8/Assign_22Assignmain/q/dense/biassave_8/RestoreV2:22*
use_locking(*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?*
validate_shape(*
T0
?
save_8/Assign_23Assignmain/q/dense/bias/Adamsave_8/RestoreV2:23*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias
?
save_8/Assign_24Assignmain/q/dense/bias/Adam_1save_8/RestoreV2:24*
validate_shape(*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
use_locking(*
T0
?
save_8/Assign_25Assignmain/q/dense/kernelsave_8/RestoreV2:25*
use_locking(*
T0*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	w?
?
save_8/Assign_26Assignmain/q/dense/kernel/Adamsave_8/RestoreV2:26*&
_class
loc:@main/q/dense/kernel*
T0*
_output_shapes
:	w?*
validate_shape(*
use_locking(
?
save_8/Assign_27Assignmain/q/dense/kernel/Adam_1save_8/RestoreV2:27*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?*
T0
?
save_8/Assign_28Assignmain/q/dense_1/biassave_8/RestoreV2:28*
_output_shapes	
:?*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
T0*
use_locking(
?
save_8/Assign_29Assignmain/q/dense_1/bias/Adamsave_8/RestoreV2:29*
T0*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save_8/Assign_30Assignmain/q/dense_1/bias/Adam_1save_8/RestoreV2:30*
validate_shape(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
use_locking(*
T0
?
save_8/Assign_31Assignmain/q/dense_1/kernelsave_8/RestoreV2:31*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_1/kernel*
T0* 
_output_shapes
:
??
?
save_8/Assign_32Assignmain/q/dense_1/kernel/Adamsave_8/RestoreV2:32*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
??*
use_locking(*
T0*
validate_shape(
?
save_8/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_8/RestoreV2:33* 
_output_shapes
:
??*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel*
T0*
use_locking(
?
save_8/Assign_34Assignmain/q/dense_2/biassave_8/RestoreV2:34*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias
?
save_8/Assign_35Assignmain/q/dense_2/bias/Adamsave_8/RestoreV2:35*
T0*
validate_shape(*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
_output_shapes
:
?
save_8/Assign_36Assignmain/q/dense_2/bias/Adam_1save_8/RestoreV2:36*
_output_shapes
:*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
T0
?
save_8/Assign_37Assignmain/q/dense_2/kernelsave_8/RestoreV2:37*(
_class
loc:@main/q/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	?
?
save_8/Assign_38Assignmain/q/dense_2/kernel/Adamsave_8/RestoreV2:38*
_output_shapes
:	?*
validate_shape(*
use_locking(*
T0*(
_class
loc:@main/q/dense_2/kernel
?
save_8/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_8/RestoreV2:39*(
_class
loc:@main/q/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	?*
validate_shape(
?
save_8/Assign_40Assigntarget/pi/dense/biassave_8/RestoreV2:40*
_output_shapes	
:?*
use_locking(*'
_class
loc:@target/pi/dense/bias*
T0*
validate_shape(
?
save_8/Assign_41Assigntarget/pi/dense/kernelsave_8/RestoreV2:41*
T0*
_output_shapes
:	o?*
use_locking(*)
_class
loc:@target/pi/dense/kernel*
validate_shape(
?
save_8/Assign_42Assigntarget/pi/dense_1/biassave_8/RestoreV2:42*)
_class
loc:@target/pi/dense_1/bias*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(
?
save_8/Assign_43Assigntarget/pi/dense_1/kernelsave_8/RestoreV2:43*
use_locking(*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
T0
?
save_8/Assign_44Assigntarget/pi/dense_2/biassave_8/RestoreV2:44*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(
?
save_8/Assign_45Assigntarget/pi/dense_2/kernelsave_8/RestoreV2:45*
_output_shapes
:	?*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
validate_shape(
?
save_8/Assign_46Assigntarget/q/dense/biassave_8/RestoreV2:46*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(*&
_class
loc:@target/q/dense/bias
?
save_8/Assign_47Assigntarget/q/dense/kernelsave_8/RestoreV2:47*
_output_shapes
:	w?*
validate_shape(*(
_class
loc:@target/q/dense/kernel*
use_locking(*
T0
?
save_8/Assign_48Assigntarget/q/dense_1/biassave_8/RestoreV2:48*
_output_shapes	
:?*
validate_shape(*
use_locking(*(
_class
loc:@target/q/dense_1/bias*
T0
?
save_8/Assign_49Assigntarget/q/dense_1/kernelsave_8/RestoreV2:49*
T0* 
_output_shapes
:
??**
_class 
loc:@target/q/dense_1/kernel*
validate_shape(*
use_locking(
?
save_8/Assign_50Assigntarget/q/dense_2/biassave_8/RestoreV2:50*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*(
_class
loc:@target/q/dense_2/bias
?
save_8/Assign_51Assigntarget/q/dense_2/kernelsave_8/RestoreV2:51*
use_locking(*
validate_shape(**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:	?*
T0
?
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_43^save_8/Assign_44^save_8/Assign_45^save_8/Assign_46^save_8/Assign_47^save_8/Assign_48^save_8/Assign_49^save_8/Assign_5^save_8/Assign_50^save_8/Assign_51^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
shape: *
_output_shapes
: *
dtype0
?
save_9/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_dbf97c145e344c6e95027f2d61a5a9b0/part*
_output_shapes
: 
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_9/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_9/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
?
save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
?

save_9/SaveV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
?
save_9/SaveV2/shape_and_slicesConst*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*)
_class
loc:@save_9/ShardedFilename*
_output_shapes
: *
T0
?
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
_output_shapes
:*

axis *
T0*
N
?
save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(
?
save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
?

save_9/RestoreV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
?
!save_9/RestoreV2/shape_and_slicesConst*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
T0*
use_locking(*
validate_shape(
?
save_9/Assign_1Assignbeta1_power_1save_9/RestoreV2:1*$
_class
loc:@main/q/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
?
save_9/Assign_2Assignbeta2_powersave_9/RestoreV2:2*
validate_shape(*
T0*%
_class
loc:@main/pi/dense/bias*
use_locking(*
_output_shapes
: 
?
save_9/Assign_3Assignbeta2_power_1save_9/RestoreV2:3*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*$
_class
loc:@main/q/dense/bias
?
save_9/Assign_4Assignmain/pi/dense/biassave_9/RestoreV2:4*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
save_9/Assign_5Assignmain/pi/dense/bias/Adamsave_9/RestoreV2:5*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes	
:?*
use_locking(
?
save_9/Assign_6Assignmain/pi/dense/bias/Adam_1save_9/RestoreV2:6*
validate_shape(*
T0*%
_class
loc:@main/pi/dense/bias*
use_locking(*
_output_shapes	
:?
?
save_9/Assign_7Assignmain/pi/dense/kernelsave_9/RestoreV2:7*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	o?*
T0
?
save_9/Assign_8Assignmain/pi/dense/kernel/Adamsave_9/RestoreV2:8*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
T0*
_output_shapes
:	o?*
use_locking(
?
save_9/Assign_9Assignmain/pi/dense/kernel/Adam_1save_9/RestoreV2:9*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
use_locking(*
T0*
validate_shape(
?
save_9/Assign_10Assignmain/pi/dense_1/biassave_9/RestoreV2:10*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?*
T0*
use_locking(*
validate_shape(
?
save_9/Assign_11Assignmain/pi/dense_1/bias/Adamsave_9/RestoreV2:11*
validate_shape(*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
T0
?
save_9/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_9/RestoreV2:12*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
save_9/Assign_13Assignmain/pi/dense_1/kernelsave_9/RestoreV2:13*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save_9/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_9/RestoreV2:14*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
??*
T0*
use_locking(
?
save_9/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_9/RestoreV2:15*
validate_shape(* 
_output_shapes
:
??*
T0*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel
?
save_9/Assign_16Assignmain/pi/dense_2/biassave_9/RestoreV2:16*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias
?
save_9/Assign_17Assignmain/pi/dense_2/bias/Adamsave_9/RestoreV2:17*
T0*
validate_shape(*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
use_locking(
?
save_9/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_9/RestoreV2:18*
T0*
_output_shapes
:*
use_locking(*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(
?
save_9/Assign_19Assignmain/pi/dense_2/kernelsave_9/RestoreV2:19*
_output_shapes
:	?*
T0*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
use_locking(
?
save_9/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_9/RestoreV2:20*
use_locking(*
validate_shape(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?
?
save_9/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_9/RestoreV2:21*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel
?
save_9/Assign_22Assignmain/q/dense/biassave_9/RestoreV2:22*
_output_shapes	
:?*
validate_shape(*
T0*$
_class
loc:@main/q/dense/bias*
use_locking(
?
save_9/Assign_23Assignmain/q/dense/bias/Adamsave_9/RestoreV2:23*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(*$
_class
loc:@main/q/dense/bias
?
save_9/Assign_24Assignmain/q/dense/bias/Adam_1save_9/RestoreV2:24*
_output_shapes	
:?*
validate_shape(*$
_class
loc:@main/q/dense/bias*
use_locking(*
T0
?
save_9/Assign_25Assignmain/q/dense/kernelsave_9/RestoreV2:25*
use_locking(*
validate_shape(*
T0*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?
?
save_9/Assign_26Assignmain/q/dense/kernel/Adamsave_9/RestoreV2:26*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
use_locking(*
T0*
validate_shape(
?
save_9/Assign_27Assignmain/q/dense/kernel/Adam_1save_9/RestoreV2:27*
_output_shapes
:	w?*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
T0
?
save_9/Assign_28Assignmain/q/dense_1/biassave_9/RestoreV2:28*
validate_shape(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
use_locking(*
T0
?
save_9/Assign_29Assignmain/q/dense_1/bias/Adamsave_9/RestoreV2:29*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
T0*
_output_shapes	
:?*
use_locking(
?
save_9/Assign_30Assignmain/q/dense_1/bias/Adam_1save_9/RestoreV2:30*
_output_shapes	
:?*
T0*&
_class
loc:@main/q/dense_1/bias*
use_locking(*
validate_shape(
?
save_9/Assign_31Assignmain/q/dense_1/kernelsave_9/RestoreV2:31*
T0*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
use_locking(
?
save_9/Assign_32Assignmain/q/dense_1/kernel/Adamsave_9/RestoreV2:32* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
use_locking(*
validate_shape(*
T0
?
save_9/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_9/RestoreV2:33*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel
?
save_9/Assign_34Assignmain/q/dense_2/biassave_9/RestoreV2:34*
validate_shape(*&
_class
loc:@main/q/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
?
save_9/Assign_35Assignmain/q/dense_2/bias/Adamsave_9/RestoreV2:35*
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(
?
save_9/Assign_36Assignmain/q/dense_2/bias/Adam_1save_9/RestoreV2:36*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense_2/bias
?
save_9/Assign_37Assignmain/q/dense_2/kernelsave_9/RestoreV2:37*
validate_shape(*
_output_shapes
:	?*
T0*
use_locking(*(
_class
loc:@main/q/dense_2/kernel
?
save_9/Assign_38Assignmain/q/dense_2/kernel/Adamsave_9/RestoreV2:38*(
_class
loc:@main/q/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	?
?
save_9/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_9/RestoreV2:39*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel*
T0*
_output_shapes
:	?*
use_locking(
?
save_9/Assign_40Assigntarget/pi/dense/biassave_9/RestoreV2:40*
_output_shapes	
:?*
validate_shape(*'
_class
loc:@target/pi/dense/bias*
use_locking(*
T0
?
save_9/Assign_41Assigntarget/pi/dense/kernelsave_9/RestoreV2:41*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?*
use_locking(*
validate_shape(
?
save_9/Assign_42Assigntarget/pi/dense_1/biassave_9/RestoreV2:42*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0*)
_class
loc:@target/pi/dense_1/bias
?
save_9/Assign_43Assigntarget/pi/dense_1/kernelsave_9/RestoreV2:43*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
??*+
_class!
loc:@target/pi/dense_1/kernel
?
save_9/Assign_44Assigntarget/pi/dense_2/biassave_9/RestoreV2:44*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias
?
save_9/Assign_45Assigntarget/pi/dense_2/kernelsave_9/RestoreV2:45*
validate_shape(*
T0*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	?
?
save_9/Assign_46Assigntarget/q/dense/biassave_9/RestoreV2:46*
validate_shape(*&
_class
loc:@target/q/dense/bias*
T0*
use_locking(*
_output_shapes	
:?
?
save_9/Assign_47Assigntarget/q/dense/kernelsave_9/RestoreV2:47*(
_class
loc:@target/q/dense/kernel*
validate_shape(*
_output_shapes
:	w?*
T0*
use_locking(
?
save_9/Assign_48Assigntarget/q/dense_1/biassave_9/RestoreV2:48*
use_locking(*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
T0
?
save_9/Assign_49Assigntarget/q/dense_1/kernelsave_9/RestoreV2:49*
use_locking(**
_class 
loc:@target/q/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:
??
?
save_9/Assign_50Assigntarget/q/dense_2/biassave_9/RestoreV2:50*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*(
_class
loc:@target/q/dense_2/bias
?
save_9/Assign_51Assigntarget/q/dense_2/kernelsave_9/RestoreV2:51**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	?
?
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_43^save_9/Assign_44^save_9/Assign_45^save_9/Assign_46^save_9/Assign_47^save_9/Assign_48^save_9/Assign_49^save_9/Assign_5^save_9/Assign_50^save_9/Assign_51^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
dtype0*
shape: *
_output_shapes
: 
?
save_10/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_191248c159b74db8a1f0641fabbd12aa/part
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_10/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
?
save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
?

save_10/SaveV2/tensor_namesConst*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0
?
save_10/SaveV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4*
dtype0
?
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_10/ShardedFilename
?
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
_output_shapes
:*

axis *
N*
T0
?
save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(
?
save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
_output_shapes
: *
T0
?

save_10/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
"save_10/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4*
dtype0
?
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_10/AssignAssignbeta1_powersave_10/RestoreV2*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
T0*
_output_shapes
: *
use_locking(
?
save_10/Assign_1Assignbeta1_power_1save_10/RestoreV2:1*$
_class
loc:@main/q/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
?
save_10/Assign_2Assignbeta2_powersave_10/RestoreV2:2*
_output_shapes
: *
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
use_locking(
?
save_10/Assign_3Assignbeta2_power_1save_10/RestoreV2:3*$
_class
loc:@main/q/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
?
save_10/Assign_4Assignmain/pi/dense/biassave_10/RestoreV2:4*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*%
_class
loc:@main/pi/dense/bias
?
save_10/Assign_5Assignmain/pi/dense/bias/Adamsave_10/RestoreV2:5*
validate_shape(*
_output_shapes	
:?*
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0
?
save_10/Assign_6Assignmain/pi/dense/bias/Adam_1save_10/RestoreV2:6*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(*%
_class
loc:@main/pi/dense/bias
?
save_10/Assign_7Assignmain/pi/dense/kernelsave_10/RestoreV2:7*
_output_shapes
:	o?*
T0*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
validate_shape(
?
save_10/Assign_8Assignmain/pi/dense/kernel/Adamsave_10/RestoreV2:8*
T0*
_output_shapes
:	o?*
validate_shape(*
use_locking(*'
_class
loc:@main/pi/dense/kernel
?
save_10/Assign_9Assignmain/pi/dense/kernel/Adam_1save_10/RestoreV2:9*
_output_shapes
:	o?*
validate_shape(*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
T0
?
save_10/Assign_10Assignmain/pi/dense_1/biassave_10/RestoreV2:10*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_10/Assign_11Assignmain/pi/dense_1/bias/Adamsave_10/RestoreV2:11*
use_locking(*'
_class
loc:@main/pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
save_10/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_10/RestoreV2:12*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:?
?
save_10/Assign_13Assignmain/pi/dense_1/kernelsave_10/RestoreV2:13*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
use_locking(*
T0
?
save_10/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_10/RestoreV2:14*
T0*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_10/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_10/RestoreV2:15*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
??*
T0
?
save_10/Assign_16Assignmain/pi/dense_2/biassave_10/RestoreV2:16*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*'
_class
loc:@main/pi/dense_2/bias
?
save_10/Assign_17Assignmain/pi/dense_2/bias/Adamsave_10/RestoreV2:17*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias
?
save_10/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_10/RestoreV2:18*
use_locking(*'
_class
loc:@main/pi/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
?
save_10/Assign_19Assignmain/pi/dense_2/kernelsave_10/RestoreV2:19*
validate_shape(*
_output_shapes
:	?*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
T0
?
save_10/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_10/RestoreV2:20*
_output_shapes
:	?*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
use_locking(
?
save_10/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_10/RestoreV2:21*
_output_shapes
:	?*
T0*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(
?
save_10/Assign_22Assignmain/q/dense/biassave_10/RestoreV2:22*
validate_shape(*
use_locking(*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
T0
?
save_10/Assign_23Assignmain/q/dense/bias/Adamsave_10/RestoreV2:23*
use_locking(*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes	
:?*
validate_shape(
?
save_10/Assign_24Assignmain/q/dense/bias/Adam_1save_10/RestoreV2:24*$
_class
loc:@main/q/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
save_10/Assign_25Assignmain/q/dense/kernelsave_10/RestoreV2:25*
validate_shape(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
T0*
use_locking(
?
save_10/Assign_26Assignmain/q/dense/kernel/Adamsave_10/RestoreV2:26*
use_locking(*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	w?*
T0
?
save_10/Assign_27Assignmain/q/dense/kernel/Adam_1save_10/RestoreV2:27*
T0*
use_locking(*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?*
validate_shape(
?
save_10/Assign_28Assignmain/q/dense_1/biassave_10/RestoreV2:28*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(*&
_class
loc:@main/q/dense_1/bias
?
save_10/Assign_29Assignmain/q/dense_1/bias/Adamsave_10/RestoreV2:29*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias
?
save_10/Assign_30Assignmain/q/dense_1/bias/Adam_1save_10/RestoreV2:30*
T0*
use_locking(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
validate_shape(
?
save_10/Assign_31Assignmain/q/dense_1/kernelsave_10/RestoreV2:31*
use_locking(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(*
T0
?
save_10/Assign_32Assignmain/q/dense_1/kernel/Adamsave_10/RestoreV2:32* 
_output_shapes
:
??*
validate_shape(*
T0*(
_class
loc:@main/q/dense_1/kernel*
use_locking(
?
save_10/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_10/RestoreV2:33*
use_locking(*
validate_shape(* 
_output_shapes
:
??*
T0*(
_class
loc:@main/q/dense_1/kernel
?
save_10/Assign_34Assignmain/q/dense_2/biassave_10/RestoreV2:34*
_output_shapes
:*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
T0
?
save_10/Assign_35Assignmain/q/dense_2/bias/Adamsave_10/RestoreV2:35*
T0*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
?
save_10/Assign_36Assignmain/q/dense_2/bias/Adam_1save_10/RestoreV2:36*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
?
save_10/Assign_37Assignmain/q/dense_2/kernelsave_10/RestoreV2:37*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	?
?
save_10/Assign_38Assignmain/q/dense_2/kernel/Adamsave_10/RestoreV2:38*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	?
?
save_10/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_10/RestoreV2:39*(
_class
loc:@main/q/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	?*
validate_shape(
?
save_10/Assign_40Assigntarget/pi/dense/biassave_10/RestoreV2:40*
T0*
validate_shape(*'
_class
loc:@target/pi/dense/bias*
use_locking(*
_output_shapes	
:?
?
save_10/Assign_41Assigntarget/pi/dense/kernelsave_10/RestoreV2:41*
use_locking(*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?*
validate_shape(*
T0
?
save_10/Assign_42Assigntarget/pi/dense_1/biassave_10/RestoreV2:42*
validate_shape(*)
_class
loc:@target/pi/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:?
?
save_10/Assign_43Assigntarget/pi/dense_1/kernelsave_10/RestoreV2:43*
validate_shape(*+
_class!
loc:@target/pi/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
??
?
save_10/Assign_44Assigntarget/pi/dense_2/biassave_10/RestoreV2:44*)
_class
loc:@target/pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
?
save_10/Assign_45Assigntarget/pi/dense_2/kernelsave_10/RestoreV2:45*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel
?
save_10/Assign_46Assigntarget/q/dense/biassave_10/RestoreV2:46*&
_class
loc:@target/q/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?
?
save_10/Assign_47Assigntarget/q/dense/kernelsave_10/RestoreV2:47*
_output_shapes
:	w?*
use_locking(*
T0*
validate_shape(*(
_class
loc:@target/q/dense/kernel
?
save_10/Assign_48Assigntarget/q/dense_1/biassave_10/RestoreV2:48*(
_class
loc:@target/q/dense_1/bias*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(
?
save_10/Assign_49Assigntarget/q/dense_1/kernelsave_10/RestoreV2:49*
T0*
validate_shape(* 
_output_shapes
:
??*
use_locking(**
_class 
loc:@target/q/dense_1/kernel
?
save_10/Assign_50Assigntarget/q/dense_2/biassave_10/RestoreV2:50*
_output_shapes
:*
validate_shape(*
T0*(
_class
loc:@target/q/dense_2/bias*
use_locking(
?
save_10/Assign_51Assigntarget/q/dense_2/kernelsave_10/RestoreV2:51*
_output_shapes
:	?*
T0*
validate_shape(**
_class 
loc:@target/q/dense_2/kernel*
use_locking(
?
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_40^save_10/Assign_41^save_10/Assign_42^save_10/Assign_43^save_10/Assign_44^save_10/Assign_45^save_10/Assign_46^save_10/Assign_47^save_10/Assign_48^save_10/Assign_49^save_10/Assign_5^save_10/Assign_50^save_10/Assign_51^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
dtype0*
shape: *
_output_shapes
: 
?
save_11/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_cf6c2af24a0a4545baa4083025366e34/part
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_11/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_11/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
?
save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
?

save_11/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
save_11/SaveV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4*
dtype0
?
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
_output_shapes
: **
_class 
loc:@save_11/ShardedFilename*
T0
?
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*

axis *
_output_shapes
:*
T0*
N
?
save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(
?
save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
T0*
_output_shapes
: 
?

save_11/RestoreV2/tensor_namesConst*
_output_shapes
:4*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
"save_11/RestoreV2/shape_and_slicesConst*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*B
dtypes8
624*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save_11/AssignAssignbeta1_powersave_11/RestoreV2*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*%
_class
loc:@main/pi/dense/bias
?
save_11/Assign_1Assignbeta1_power_1save_11/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@main/q/dense/bias
?
save_11/Assign_2Assignbeta2_powersave_11/RestoreV2:2*
T0*
use_locking(*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
?
save_11/Assign_3Assignbeta2_power_1save_11/RestoreV2:3*
T0*
use_locking(*
_output_shapes
: *$
_class
loc:@main/q/dense/bias*
validate_shape(
?
save_11/Assign_4Assignmain/pi/dense/biassave_11/RestoreV2:4*
use_locking(*
_output_shapes	
:?*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(
?
save_11/Assign_5Assignmain/pi/dense/bias/Adamsave_11/RestoreV2:5*
T0*
use_locking(*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?
?
save_11/Assign_6Assignmain/pi/dense/bias/Adam_1save_11/RestoreV2:6*
validate_shape(*
T0*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
use_locking(
?
save_11/Assign_7Assignmain/pi/dense/kernelsave_11/RestoreV2:7*
validate_shape(*
T0*
_output_shapes
:	o?*
use_locking(*'
_class
loc:@main/pi/dense/kernel
?
save_11/Assign_8Assignmain/pi/dense/kernel/Adamsave_11/RestoreV2:8*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
T0*
use_locking(
?
save_11/Assign_9Assignmain/pi/dense/kernel/Adam_1save_11/RestoreV2:9*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	o?*
T0
?
save_11/Assign_10Assignmain/pi/dense_1/biassave_11/RestoreV2:10*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_11/Assign_11Assignmain/pi/dense_1/bias/Adamsave_11/RestoreV2:11*
T0*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
save_11/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_11/RestoreV2:12*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
T0*
validate_shape(*
use_locking(
?
save_11/Assign_13Assignmain/pi/dense_1/kernelsave_11/RestoreV2:13*
validate_shape(* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(*
T0
?
save_11/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_11/RestoreV2:14* 
_output_shapes
:
??*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
T0
?
save_11/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_11/RestoreV2:15*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_11/Assign_16Assignmain/pi/dense_2/biassave_11/RestoreV2:16*'
_class
loc:@main/pi/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
?
save_11/Assign_17Assignmain/pi/dense_2/bias/Adamsave_11/RestoreV2:17*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
?
save_11/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_11/RestoreV2:18*
T0*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
_output_shapes
:
?
save_11/Assign_19Assignmain/pi/dense_2/kernelsave_11/RestoreV2:19*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel
?
save_11/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_11/RestoreV2:20*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(
?
save_11/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_11/RestoreV2:21*
_output_shapes
:	?*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
use_locking(
?
save_11/Assign_22Assignmain/q/dense/biassave_11/RestoreV2:22*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(*$
_class
loc:@main/q/dense/bias
?
save_11/Assign_23Assignmain/q/dense/bias/Adamsave_11/RestoreV2:23*
_output_shapes	
:?*
use_locking(*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(
?
save_11/Assign_24Assignmain/q/dense/bias/Adam_1save_11/RestoreV2:24*
_output_shapes	
:?*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(*
use_locking(
?
save_11/Assign_25Assignmain/q/dense/kernelsave_11/RestoreV2:25*
_output_shapes
:	w?*
use_locking(*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
T0
?
save_11/Assign_26Assignmain/q/dense/kernel/Adamsave_11/RestoreV2:26*
validate_shape(*
use_locking(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
T0
?
save_11/Assign_27Assignmain/q/dense/kernel/Adam_1save_11/RestoreV2:27*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
T0*
validate_shape(*
use_locking(
?
save_11/Assign_28Assignmain/q/dense_1/biassave_11/RestoreV2:28*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0*&
_class
loc:@main/q/dense_1/bias
?
save_11/Assign_29Assignmain/q/dense_1/bias/Adamsave_11/RestoreV2:29*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
save_11/Assign_30Assignmain/q/dense_1/bias/Adam_1save_11/RestoreV2:30*
use_locking(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
save_11/Assign_31Assignmain/q/dense_1/kernelsave_11/RestoreV2:31*
validate_shape(*
T0* 
_output_shapes
:
??*
use_locking(*(
_class
loc:@main/q/dense_1/kernel
?
save_11/Assign_32Assignmain/q/dense_1/kernel/Adamsave_11/RestoreV2:32*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
??
?
save_11/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_11/RestoreV2:33*(
_class
loc:@main/q/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
??
?
save_11/Assign_34Assignmain/q/dense_2/biassave_11/RestoreV2:34*
validate_shape(*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
?
save_11/Assign_35Assignmain/q/dense_2/bias/Adamsave_11/RestoreV2:35*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias
?
save_11/Assign_36Assignmain/q/dense_2/bias/Adam_1save_11/RestoreV2:36*
T0*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
?
save_11/Assign_37Assignmain/q/dense_2/kernelsave_11/RestoreV2:37*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
T0*
_output_shapes
:	?
?
save_11/Assign_38Assignmain/q/dense_2/kernel/Adamsave_11/RestoreV2:38*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel
?
save_11/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_11/RestoreV2:39*
T0*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
use_locking(*
validate_shape(
?
save_11/Assign_40Assigntarget/pi/dense/biassave_11/RestoreV2:40*
use_locking(*
validate_shape(*
_output_shapes	
:?*'
_class
loc:@target/pi/dense/bias*
T0
?
save_11/Assign_41Assigntarget/pi/dense/kernelsave_11/RestoreV2:41*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?*
use_locking(*
validate_shape(
?
save_11/Assign_42Assigntarget/pi/dense_1/biassave_11/RestoreV2:42*
validate_shape(*)
_class
loc:@target/pi/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:?
?
save_11/Assign_43Assigntarget/pi/dense_1/kernelsave_11/RestoreV2:43*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
??
?
save_11/Assign_44Assigntarget/pi/dense_2/biassave_11/RestoreV2:44*
T0*
validate_shape(*)
_class
loc:@target/pi/dense_2/bias*
use_locking(*
_output_shapes
:
?
save_11/Assign_45Assigntarget/pi/dense_2/kernelsave_11/RestoreV2:45*
validate_shape(*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	?*
use_locking(
?
save_11/Assign_46Assigntarget/q/dense/biassave_11/RestoreV2:46*
use_locking(*
_output_shapes	
:?*&
_class
loc:@target/q/dense/bias*
T0*
validate_shape(
?
save_11/Assign_47Assigntarget/q/dense/kernelsave_11/RestoreV2:47*
T0*
_output_shapes
:	w?*(
_class
loc:@target/q/dense/kernel*
use_locking(*
validate_shape(
?
save_11/Assign_48Assigntarget/q/dense_1/biassave_11/RestoreV2:48*
validate_shape(*(
_class
loc:@target/q/dense_1/bias*
use_locking(*
_output_shapes	
:?*
T0
?
save_11/Assign_49Assigntarget/q/dense_1/kernelsave_11/RestoreV2:49* 
_output_shapes
:
??**
_class 
loc:@target/q/dense_1/kernel*
use_locking(*
validate_shape(*
T0
?
save_11/Assign_50Assigntarget/q/dense_2/biassave_11/RestoreV2:50*
T0*
_output_shapes
:*(
_class
loc:@target/q/dense_2/bias*
use_locking(*
validate_shape(
?
save_11/Assign_51Assigntarget/q/dense_2/kernelsave_11/RestoreV2:51*
validate_shape(*
_output_shapes
:	?*
T0*
use_locking(**
_class 
loc:@target/q/dense_2/kernel
?
save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_20^save_11/Assign_21^save_11/Assign_22^save_11/Assign_23^save_11/Assign_24^save_11/Assign_25^save_11/Assign_26^save_11/Assign_27^save_11/Assign_28^save_11/Assign_29^save_11/Assign_3^save_11/Assign_30^save_11/Assign_31^save_11/Assign_32^save_11/Assign_33^save_11/Assign_34^save_11/Assign_35^save_11/Assign_36^save_11/Assign_37^save_11/Assign_38^save_11/Assign_39^save_11/Assign_4^save_11/Assign_40^save_11/Assign_41^save_11/Assign_42^save_11/Assign_43^save_11/Assign_44^save_11/Assign_45^save_11/Assign_46^save_11/Assign_47^save_11/Assign_48^save_11/Assign_49^save_11/Assign_5^save_11/Assign_50^save_11/Assign_51^save_11/Assign_6^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard
\
save_12/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
shape: *
dtype0*
_output_shapes
: 
?
save_12/StringJoin/inputs_1Const*<
value3B1 B+_temp_1b4be7c28c1d48228918333622fddf28/part*
_output_shapes
: *
dtype0
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_12/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_12/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
?
save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
?

save_12/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
save_12/SaveV2/shape_and_slicesConst*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_12/ShardedFilename
?
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*
_output_shapes
:*
N*

axis *
T0
?
save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(
?
save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
T0*
_output_shapes
: 
?

save_12/RestoreV2/tensor_namesConst*
_output_shapes
:4*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
"save_12/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
?
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_12/AssignAssignbeta1_powersave_12/RestoreV2*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
?
save_12/Assign_1Assignbeta1_power_1save_12/RestoreV2:1*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*$
_class
loc:@main/q/dense/bias
?
save_12/Assign_2Assignbeta2_powersave_12/RestoreV2:2*
use_locking(*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
validate_shape(*
T0
?
save_12/Assign_3Assignbeta2_power_1save_12/RestoreV2:3*$
_class
loc:@main/q/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
?
save_12/Assign_4Assignmain/pi/dense/biassave_12/RestoreV2:4*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(*%
_class
loc:@main/pi/dense/bias
?
save_12/Assign_5Assignmain/pi/dense/bias/Adamsave_12/RestoreV2:5*
T0*
_output_shapes	
:?*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
use_locking(
?
save_12/Assign_6Assignmain/pi/dense/bias/Adam_1save_12/RestoreV2:6*
T0*
use_locking(*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
validate_shape(
?
save_12/Assign_7Assignmain/pi/dense/kernelsave_12/RestoreV2:7*
_output_shapes
:	o?*
T0*
validate_shape(*
use_locking(*'
_class
loc:@main/pi/dense/kernel
?
save_12/Assign_8Assignmain/pi/dense/kernel/Adamsave_12/RestoreV2:8*
use_locking(*
T0*
_output_shapes
:	o?*
validate_shape(*'
_class
loc:@main/pi/dense/kernel
?
save_12/Assign_9Assignmain/pi/dense/kernel/Adam_1save_12/RestoreV2:9*
_output_shapes
:	o?*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
T0*
use_locking(
?
save_12/Assign_10Assignmain/pi/dense_1/biassave_12/RestoreV2:10*
validate_shape(*
T0*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?*
use_locking(
?
save_12/Assign_11Assignmain/pi/dense_1/bias/Adamsave_12/RestoreV2:11*
T0*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
validate_shape(
?
save_12/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_12/RestoreV2:12*
use_locking(*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?*
validate_shape(*
T0
?
save_12/Assign_13Assignmain/pi/dense_1/kernelsave_12/RestoreV2:13*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
T0* 
_output_shapes
:
??
?
save_12/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_12/RestoreV2:14*
T0*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(* 
_output_shapes
:
??
?
save_12/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_12/RestoreV2:15*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:
??
?
save_12/Assign_16Assignmain/pi/dense_2/biassave_12/RestoreV2:16*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
?
save_12/Assign_17Assignmain/pi/dense_2/bias/Adamsave_12/RestoreV2:17*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
?
save_12/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_12/RestoreV2:18*
T0*
_output_shapes
:*
use_locking(*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(
?
save_12/Assign_19Assignmain/pi/dense_2/kernelsave_12/RestoreV2:19*
_output_shapes
:	?*
T0*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
use_locking(
?
save_12/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_12/RestoreV2:20*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
validate_shape(
?
save_12/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_12/RestoreV2:21*
_output_shapes
:	?*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
T0
?
save_12/Assign_22Assignmain/q/dense/biassave_12/RestoreV2:22*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(*$
_class
loc:@main/q/dense/bias
?
save_12/Assign_23Assignmain/q/dense/bias/Adamsave_12/RestoreV2:23*
_output_shapes	
:?*
use_locking(*$
_class
loc:@main/q/dense/bias*
T0*
validate_shape(
?
save_12/Assign_24Assignmain/q/dense/bias/Adam_1save_12/RestoreV2:24*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(
?
save_12/Assign_25Assignmain/q/dense/kernelsave_12/RestoreV2:25*
use_locking(*&
_class
loc:@main/q/dense/kernel*
T0*
_output_shapes
:	w?*
validate_shape(
?
save_12/Assign_26Assignmain/q/dense/kernel/Adamsave_12/RestoreV2:26*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
T0*
validate_shape(*
use_locking(
?
save_12/Assign_27Assignmain/q/dense/kernel/Adam_1save_12/RestoreV2:27*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?*
T0
?
save_12/Assign_28Assignmain/q/dense_1/biassave_12/RestoreV2:28*
T0*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?
?
save_12/Assign_29Assignmain/q/dense_1/bias/Adamsave_12/RestoreV2:29*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(*&
_class
loc:@main/q/dense_1/bias
?
save_12/Assign_30Assignmain/q/dense_1/bias/Adam_1save_12/RestoreV2:30*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*&
_class
loc:@main/q/dense_1/bias
?
save_12/Assign_31Assignmain/q/dense_1/kernelsave_12/RestoreV2:31*
T0*(
_class
loc:@main/q/dense_1/kernel*
use_locking(*
validate_shape(* 
_output_shapes
:
??
?
save_12/Assign_32Assignmain/q/dense_1/kernel/Adamsave_12/RestoreV2:32*
use_locking(*
T0* 
_output_shapes
:
??*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel
?
save_12/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_12/RestoreV2:33*
validate_shape(*
use_locking(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
T0
?
save_12/Assign_34Assignmain/q/dense_2/biassave_12/RestoreV2:34*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias
?
save_12/Assign_35Assignmain/q/dense_2/bias/Adamsave_12/RestoreV2:35*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
?
save_12/Assign_36Assignmain/q/dense_2/bias/Adam_1save_12/RestoreV2:36*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
?
save_12/Assign_37Assignmain/q/dense_2/kernelsave_12/RestoreV2:37*
validate_shape(*
T0*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	?
?
save_12/Assign_38Assignmain/q/dense_2/kernel/Adamsave_12/RestoreV2:38*(
_class
loc:@main/q/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(
?
save_12/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_12/RestoreV2:39*
T0*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	?*
use_locking(
?
save_12/Assign_40Assigntarget/pi/dense/biassave_12/RestoreV2:40*
_output_shapes	
:?*
T0*
use_locking(*'
_class
loc:@target/pi/dense/bias*
validate_shape(
?
save_12/Assign_41Assigntarget/pi/dense/kernelsave_12/RestoreV2:41*
validate_shape(*
_output_shapes
:	o?*)
_class
loc:@target/pi/dense/kernel*
use_locking(*
T0
?
save_12/Assign_42Assigntarget/pi/dense_1/biassave_12/RestoreV2:42*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?
?
save_12/Assign_43Assigntarget/pi/dense_1/kernelsave_12/RestoreV2:43*
use_locking(*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:
??
?
save_12/Assign_44Assigntarget/pi/dense_2/biassave_12/RestoreV2:44*)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
?
save_12/Assign_45Assigntarget/pi/dense_2/kernelsave_12/RestoreV2:45*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	?
?
save_12/Assign_46Assigntarget/q/dense/biassave_12/RestoreV2:46*
T0*
use_locking(*
validate_shape(*&
_class
loc:@target/q/dense/bias*
_output_shapes	
:?
?
save_12/Assign_47Assigntarget/q/dense/kernelsave_12/RestoreV2:47*
T0*
validate_shape(*
_output_shapes
:	w?*
use_locking(*(
_class
loc:@target/q/dense/kernel
?
save_12/Assign_48Assigntarget/q/dense_1/biassave_12/RestoreV2:48*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
T0*
use_locking(
?
save_12/Assign_49Assigntarget/q/dense_1/kernelsave_12/RestoreV2:49*
validate_shape(*
use_locking(**
_class 
loc:@target/q/dense_1/kernel*
T0* 
_output_shapes
:
??
?
save_12/Assign_50Assigntarget/q/dense_2/biassave_12/RestoreV2:50*
_output_shapes
:*
validate_shape(*
T0*(
_class
loc:@target/q/dense_2/bias*
use_locking(
?
save_12/Assign_51Assigntarget/q/dense_2/kernelsave_12/RestoreV2:51**
_class 
loc:@target/q/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	?*
validate_shape(
?
save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_38^save_12/Assign_39^save_12/Assign_4^save_12/Assign_40^save_12/Assign_41^save_12/Assign_42^save_12/Assign_43^save_12/Assign_44^save_12/Assign_45^save_12/Assign_46^save_12/Assign_47^save_12/Assign_48^save_12/Assign_49^save_12/Assign_5^save_12/Assign_50^save_12/Assign_51^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
dtype0*
_output_shapes
: *
shape: 
?
save_13/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_10cfea75aaa14ea1a2ae6b8e485f965f/part*
dtype0
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
T
save_13/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_13/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
?
save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
?

save_13/SaveV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
?
save_13/SaveV2/shape_and_slicesConst*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_13/ShardedFilename
?
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
_output_shapes
:*
N*
T0*

axis 
?
save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(
?
save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
_output_shapes
: *
T0
?

save_13/RestoreV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
"save_13/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4*
dtype0
?
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*B
dtypes8
624*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0
?
save_13/Assign_1Assignbeta1_power_1save_13/RestoreV2:1*$
_class
loc:@main/q/dense/bias*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
?
save_13/Assign_2Assignbeta2_powersave_13/RestoreV2:2*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
?
save_13/Assign_3Assignbeta2_power_1save_13/RestoreV2:3*
T0*
_output_shapes
: *$
_class
loc:@main/q/dense/bias*
validate_shape(*
use_locking(
?
save_13/Assign_4Assignmain/pi/dense/biassave_13/RestoreV2:4*
validate_shape(*
_output_shapes	
:?*
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0
?
save_13/Assign_5Assignmain/pi/dense/bias/Adamsave_13/RestoreV2:5*
use_locking(*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
save_13/Assign_6Assignmain/pi/dense/bias/Adam_1save_13/RestoreV2:6*
validate_shape(*
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes	
:?
?
save_13/Assign_7Assignmain/pi/dense/kernelsave_13/RestoreV2:7*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
T0*
use_locking(
?
save_13/Assign_8Assignmain/pi/dense/kernel/Adamsave_13/RestoreV2:8*
T0*
_output_shapes
:	o?*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
use_locking(
?
save_13/Assign_9Assignmain/pi/dense/kernel/Adam_1save_13/RestoreV2:9*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
T0*
use_locking(
?
save_13/Assign_10Assignmain/pi/dense_1/biassave_13/RestoreV2:10*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0
?
save_13/Assign_11Assignmain/pi/dense_1/bias/Adamsave_13/RestoreV2:11*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
save_13/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_13/RestoreV2:12*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0*'
_class
loc:@main/pi/dense_1/bias
?
save_13/Assign_13Assignmain/pi/dense_1/kernelsave_13/RestoreV2:13*
T0* 
_output_shapes
:
??*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel
?
save_13/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_13/RestoreV2:14*
validate_shape(*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
T0
?
save_13/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_13/RestoreV2:15* 
_output_shapes
:
??*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel*
T0*
validate_shape(
?
save_13/Assign_16Assignmain/pi/dense_2/biassave_13/RestoreV2:16*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
?
save_13/Assign_17Assignmain/pi/dense_2/bias/Adamsave_13/RestoreV2:17*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
?
save_13/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_13/RestoreV2:18*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
use_locking(*
T0
?
save_13/Assign_19Assignmain/pi/dense_2/kernelsave_13/RestoreV2:19*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
validate_shape(*
use_locking(
?
save_13/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_13/RestoreV2:20*
T0*
validate_shape(*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?
?
save_13/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_13/RestoreV2:21*
use_locking(*
validate_shape(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?
?
save_13/Assign_22Assignmain/q/dense/biassave_13/RestoreV2:22*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
validate_shape(*
use_locking(*
T0
?
save_13/Assign_23Assignmain/q/dense/bias/Adamsave_13/RestoreV2:23*
use_locking(*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes	
:?*
T0
?
save_13/Assign_24Assignmain/q/dense/bias/Adam_1save_13/RestoreV2:24*
validate_shape(*$
_class
loc:@main/q/dense/bias*
use_locking(*
_output_shapes	
:?*
T0
?
save_13/Assign_25Assignmain/q/dense/kernelsave_13/RestoreV2:25*
T0*
use_locking(*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
_output_shapes
:	w?
?
save_13/Assign_26Assignmain/q/dense/kernel/Adamsave_13/RestoreV2:26*&
_class
loc:@main/q/dense/kernel*
T0*
use_locking(*
_output_shapes
:	w?*
validate_shape(
?
save_13/Assign_27Assignmain/q/dense/kernel/Adam_1save_13/RestoreV2:27*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?*
use_locking(*
T0
?
save_13/Assign_28Assignmain/q/dense_1/biassave_13/RestoreV2:28*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(
?
save_13/Assign_29Assignmain/q/dense_1/bias/Adamsave_13/RestoreV2:29*
T0*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_13/Assign_30Assignmain/q/dense_1/bias/Adam_1save_13/RestoreV2:30*
validate_shape(*
use_locking(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
T0
?
save_13/Assign_31Assignmain/q/dense_1/kernelsave_13/RestoreV2:31* 
_output_shapes
:
??*
use_locking(*
T0*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel
?
save_13/Assign_32Assignmain/q/dense_1/kernel/Adamsave_13/RestoreV2:32*
use_locking(*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel*
T0* 
_output_shapes
:
??
?
save_13/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_13/RestoreV2:33*(
_class
loc:@main/q/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
T0*
use_locking(
?
save_13/Assign_34Assignmain/q/dense_2/biassave_13/RestoreV2:34*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
?
save_13/Assign_35Assignmain/q/dense_2/bias/Adamsave_13/RestoreV2:35*
_output_shapes
:*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
use_locking(
?
save_13/Assign_36Assignmain/q/dense_2/bias/Adam_1save_13/RestoreV2:36*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*&
_class
loc:@main/q/dense_2/bias
?
save_13/Assign_37Assignmain/q/dense_2/kernelsave_13/RestoreV2:37*
validate_shape(*
use_locking(*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
T0
?
save_13/Assign_38Assignmain/q/dense_2/kernel/Adamsave_13/RestoreV2:38*(
_class
loc:@main/q/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	?
?
save_13/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_13/RestoreV2:39*
validate_shape(*
T0*
_output_shapes
:	?*
use_locking(*(
_class
loc:@main/q/dense_2/kernel
?
save_13/Assign_40Assigntarget/pi/dense/biassave_13/RestoreV2:40*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_13/Assign_41Assigntarget/pi/dense/kernelsave_13/RestoreV2:41*
validate_shape(*
use_locking(*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?*
T0
?
save_13/Assign_42Assigntarget/pi/dense_1/biassave_13/RestoreV2:42*
_output_shapes	
:?*
use_locking(*)
_class
loc:@target/pi/dense_1/bias*
T0*
validate_shape(
?
save_13/Assign_43Assigntarget/pi/dense_1/kernelsave_13/RestoreV2:43*
T0*
validate_shape(* 
_output_shapes
:
??*+
_class!
loc:@target/pi/dense_1/kernel*
use_locking(
?
save_13/Assign_44Assigntarget/pi/dense_2/biassave_13/RestoreV2:44*
use_locking(*)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
?
save_13/Assign_45Assigntarget/pi/dense_2/kernelsave_13/RestoreV2:45*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	?
?
save_13/Assign_46Assigntarget/q/dense/biassave_13/RestoreV2:46*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(*&
_class
loc:@target/q/dense/bias
?
save_13/Assign_47Assigntarget/q/dense/kernelsave_13/RestoreV2:47*
use_locking(*
_output_shapes
:	w?*
T0*
validate_shape(*(
_class
loc:@target/q/dense/kernel
?
save_13/Assign_48Assigntarget/q/dense_1/biassave_13/RestoreV2:48*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias
?
save_13/Assign_49Assigntarget/q/dense_1/kernelsave_13/RestoreV2:49*
T0*
validate_shape(*
use_locking(**
_class 
loc:@target/q/dense_1/kernel* 
_output_shapes
:
??
?
save_13/Assign_50Assigntarget/q/dense_2/biassave_13/RestoreV2:50*
T0*
use_locking(*
_output_shapes
:*(
_class
loc:@target/q/dense_2/bias*
validate_shape(
?
save_13/Assign_51Assigntarget/q/dense_2/kernelsave_13/RestoreV2:51*
T0**
_class 
loc:@target/q/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	?
?
save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_38^save_13/Assign_39^save_13/Assign_4^save_13/Assign_40^save_13/Assign_41^save_13/Assign_42^save_13/Assign_43^save_13/Assign_44^save_13/Assign_45^save_13/Assign_46^save_13/Assign_47^save_13/Assign_48^save_13/Assign_49^save_13/Assign_5^save_13/Assign_50^save_13/Assign_51^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
dtype0*
_output_shapes
: *
shape: 
?
save_14/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_6dad41e569f941bb90fa09ef25762ab5/part*
_output_shapes
: 
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_14/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_14/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
?
save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
?

save_14/SaveV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
save_14/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2**
_class 
loc:@save_14/ShardedFilename*
_output_shapes
: *
T0
?
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*
T0*
_output_shapes
:*
N*

axis 
?
save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(
?
save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
_output_shapes
: *
T0
?

save_14/RestoreV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4*
dtype0
?
"save_14/RestoreV2/shape_and_slicesConst*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4
?
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*B
dtypes8
624*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save_14/AssignAssignbeta1_powersave_14/RestoreV2*
T0*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
validate_shape(*
use_locking(
?
save_14/Assign_1Assignbeta1_power_1save_14/RestoreV2:1*
T0*
validate_shape(*
_output_shapes
: *$
_class
loc:@main/q/dense/bias*
use_locking(
?
save_14/Assign_2Assignbeta2_powersave_14/RestoreV2:2*
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes
: *
validate_shape(
?
save_14/Assign_3Assignbeta2_power_1save_14/RestoreV2:3*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
?
save_14/Assign_4Assignmain/pi/dense/biassave_14/RestoreV2:4*%
_class
loc:@main/pi/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_14/Assign_5Assignmain/pi/dense/bias/Adamsave_14/RestoreV2:5*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias
?
save_14/Assign_6Assignmain/pi/dense/bias/Adam_1save_14/RestoreV2:6*%
_class
loc:@main/pi/dense/bias*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(
?
save_14/Assign_7Assignmain/pi/dense/kernelsave_14/RestoreV2:7*
validate_shape(*
_output_shapes
:	o?*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
T0
?
save_14/Assign_8Assignmain/pi/dense/kernel/Adamsave_14/RestoreV2:8*
validate_shape(*
T0*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
_output_shapes
:	o?
?
save_14/Assign_9Assignmain/pi/dense/kernel/Adam_1save_14/RestoreV2:9*
use_locking(*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0*
validate_shape(
?
save_14/Assign_10Assignmain/pi/dense_1/biassave_14/RestoreV2:10*
use_locking(*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:?*
validate_shape(
?
save_14/Assign_11Assignmain/pi/dense_1/bias/Adamsave_14/RestoreV2:11*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:?
?
save_14/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_14/RestoreV2:12*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
save_14/Assign_13Assignmain/pi/dense_1/kernelsave_14/RestoreV2:13*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
??
?
save_14/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_14/RestoreV2:14*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
T0*
use_locking(
?
save_14/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_14/RestoreV2:15*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
use_locking(*
T0*
validate_shape(
?
save_14/Assign_16Assignmain/pi/dense_2/biassave_14/RestoreV2:16*
_output_shapes
:*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
T0
?
save_14/Assign_17Assignmain/pi/dense_2/bias/Adamsave_14/RestoreV2:17*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias
?
save_14/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_14/RestoreV2:18*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
T0
?
save_14/Assign_19Assignmain/pi/dense_2/kernelsave_14/RestoreV2:19*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
use_locking(*
T0
?
save_14/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_14/RestoreV2:20*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
T0*
validate_shape(
?
save_14/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_14/RestoreV2:21*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	?
?
save_14/Assign_22Assignmain/q/dense/biassave_14/RestoreV2:22*
validate_shape(*
T0*
use_locking(*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?
?
save_14/Assign_23Assignmain/q/dense/bias/Adamsave_14/RestoreV2:23*
_output_shapes	
:?*
validate_shape(*$
_class
loc:@main/q/dense/bias*
use_locking(*
T0
?
save_14/Assign_24Assignmain/q/dense/bias/Adam_1save_14/RestoreV2:24*
validate_shape(*
use_locking(*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes	
:?
?
save_14/Assign_25Assignmain/q/dense/kernelsave_14/RestoreV2:25*
_output_shapes
:	w?*
validate_shape(*
T0*&
_class
loc:@main/q/dense/kernel*
use_locking(
?
save_14/Assign_26Assignmain/q/dense/kernel/Adamsave_14/RestoreV2:26*
T0*
validate_shape(*
_output_shapes
:	w?*
use_locking(*&
_class
loc:@main/q/dense/kernel
?
save_14/Assign_27Assignmain/q/dense/kernel/Adam_1save_14/RestoreV2:27*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
T0*
use_locking(
?
save_14/Assign_28Assignmain/q/dense_1/biassave_14/RestoreV2:28*&
_class
loc:@main/q/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
?
save_14/Assign_29Assignmain/q/dense_1/bias/Adamsave_14/RestoreV2:29*
use_locking(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
T0*
validate_shape(
?
save_14/Assign_30Assignmain/q/dense_1/bias/Adam_1save_14/RestoreV2:30*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0*&
_class
loc:@main/q/dense_1/bias
?
save_14/Assign_31Assignmain/q/dense_1/kernelsave_14/RestoreV2:31*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel
?
save_14/Assign_32Assignmain/q/dense_1/kernel/Adamsave_14/RestoreV2:32*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel*
T0*
use_locking(* 
_output_shapes
:
??
?
save_14/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_14/RestoreV2:33*
T0*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_14/Assign_34Assignmain/q/dense_2/biassave_14/RestoreV2:34*
use_locking(*
_output_shapes
:*
validate_shape(*&
_class
loc:@main/q/dense_2/bias*
T0
?
save_14/Assign_35Assignmain/q/dense_2/bias/Adamsave_14/RestoreV2:35*
T0*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
?
save_14/Assign_36Assignmain/q/dense_2/bias/Adam_1save_14/RestoreV2:36*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias
?
save_14/Assign_37Assignmain/q/dense_2/kernelsave_14/RestoreV2:37*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
T0*
use_locking(
?
save_14/Assign_38Assignmain/q/dense_2/kernel/Adamsave_14/RestoreV2:38*
T0*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(
?
save_14/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_14/RestoreV2:39*
use_locking(*
T0*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	?
?
save_14/Assign_40Assigntarget/pi/dense/biassave_14/RestoreV2:40*
T0*'
_class
loc:@target/pi/dense/bias*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save_14/Assign_41Assigntarget/pi/dense/kernelsave_14/RestoreV2:41*)
_class
loc:@target/pi/dense/kernel*
use_locking(*
T0*
_output_shapes
:	o?*
validate_shape(
?
save_14/Assign_42Assigntarget/pi/dense_1/biassave_14/RestoreV2:42*
T0*)
_class
loc:@target/pi/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save_14/Assign_43Assigntarget/pi/dense_1/kernelsave_14/RestoreV2:43*
validate_shape(*
use_locking(* 
_output_shapes
:
??*
T0*+
_class!
loc:@target/pi/dense_1/kernel
?
save_14/Assign_44Assigntarget/pi/dense_2/biassave_14/RestoreV2:44*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*)
_class
loc:@target/pi/dense_2/bias
?
save_14/Assign_45Assigntarget/pi/dense_2/kernelsave_14/RestoreV2:45*
use_locking(*
_output_shapes
:	?*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
validate_shape(
?
save_14/Assign_46Assigntarget/q/dense/biassave_14/RestoreV2:46*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(*&
_class
loc:@target/q/dense/bias
?
save_14/Assign_47Assigntarget/q/dense/kernelsave_14/RestoreV2:47*
T0*
use_locking(*
_output_shapes
:	w?*
validate_shape(*(
_class
loc:@target/q/dense/kernel
?
save_14/Assign_48Assigntarget/q/dense_1/biassave_14/RestoreV2:48*
_output_shapes	
:?*
use_locking(*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
T0
?
save_14/Assign_49Assigntarget/q/dense_1/kernelsave_14/RestoreV2:49*
use_locking(**
_class 
loc:@target/q/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:
??
?
save_14/Assign_50Assigntarget/q/dense_2/biassave_14/RestoreV2:50*
use_locking(*
_output_shapes
:*
validate_shape(*(
_class
loc:@target/q/dense_2/bias*
T0
?
save_14/Assign_51Assigntarget/q/dense_2/kernelsave_14/RestoreV2:51*
T0*
use_locking(*
_output_shapes
:	?**
_class 
loc:@target/q/dense_2/kernel*
validate_shape(
?
save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_38^save_14/Assign_39^save_14/Assign_4^save_14/Assign_40^save_14/Assign_41^save_14/Assign_42^save_14/Assign_43^save_14/Assign_44^save_14/Assign_45^save_14/Assign_46^save_14/Assign_47^save_14/Assign_48^save_14/Assign_49^save_14/Assign_5^save_14/Assign_50^save_14/Assign_51^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
_output_shapes
: *
dtype0*
shape: 
?
save_15/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_a991781095744d32be23046dbf144c16/part*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_15/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_15/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
?
save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
?

save_15/SaveV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
save_15/SaveV2/shape_and_slicesConst*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4
?
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2*
_output_shapes
: **
_class 
loc:@save_15/ShardedFilename*
T0
?
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*

axis *
_output_shapes
:*
N*
T0
?
save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(
?
save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
_output_shapes
: *
T0
?

save_15/RestoreV2/tensor_namesConst*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0*
_output_shapes
:4
?
"save_15/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_15/AssignAssignbeta1_powersave_15/RestoreV2*%
_class
loc:@main/pi/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
?
save_15/Assign_1Assignbeta1_power_1save_15/RestoreV2:1*
use_locking(*$
_class
loc:@main/q/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
?
save_15/Assign_2Assignbeta2_powersave_15/RestoreV2:2*
T0*
validate_shape(*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
use_locking(
?
save_15/Assign_3Assignbeta2_power_1save_15/RestoreV2:3*
use_locking(*
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias*
_output_shapes
: 
?
save_15/Assign_4Assignmain/pi/dense/biassave_15/RestoreV2:4*
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
save_15/Assign_5Assignmain/pi/dense/bias/Adamsave_15/RestoreV2:5*
T0*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?*
use_locking(
?
save_15/Assign_6Assignmain/pi/dense/bias/Adam_1save_15/RestoreV2:6*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias
?
save_15/Assign_7Assignmain/pi/dense/kernelsave_15/RestoreV2:7*
T0*
_output_shapes
:	o?*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
use_locking(
?
save_15/Assign_8Assignmain/pi/dense/kernel/Adamsave_15/RestoreV2:8*
T0*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
use_locking(*
_output_shapes
:	o?
?
save_15/Assign_9Assignmain/pi/dense/kernel/Adam_1save_15/RestoreV2:9*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
T0*
use_locking(
?
save_15/Assign_10Assignmain/pi/dense_1/biassave_15/RestoreV2:10*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0
?
save_15/Assign_11Assignmain/pi/dense_1/bias/Adamsave_15/RestoreV2:11*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:?
?
save_15/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_15/RestoreV2:12*
T0*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
_output_shapes	
:?
?
save_15/Assign_13Assignmain/pi/dense_1/kernelsave_15/RestoreV2:13*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel
?
save_15/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_15/RestoreV2:14*
use_locking(* 
_output_shapes
:
??*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
T0
?
save_15/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_15/RestoreV2:15*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel
?
save_15/Assign_16Assignmain/pi/dense_2/biassave_15/RestoreV2:16*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias
?
save_15/Assign_17Assignmain/pi/dense_2/bias/Adamsave_15/RestoreV2:17*'
_class
loc:@main/pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
?
save_15/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_15/RestoreV2:18*'
_class
loc:@main/pi/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
?
save_15/Assign_19Assignmain/pi/dense_2/kernelsave_15/RestoreV2:19*)
_class
loc:@main/pi/dense_2/kernel*
use_locking(*
_output_shapes
:	?*
T0*
validate_shape(
?
save_15/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_15/RestoreV2:20*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	?
?
save_15/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_15/RestoreV2:21*
use_locking(*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
T0
?
save_15/Assign_22Assignmain/q/dense/biassave_15/RestoreV2:22*
use_locking(*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes	
:?*
validate_shape(
?
save_15/Assign_23Assignmain/q/dense/bias/Adamsave_15/RestoreV2:23*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0
?
save_15/Assign_24Assignmain/q/dense/bias/Adam_1save_15/RestoreV2:24*$
_class
loc:@main/q/dense/bias*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(
?
save_15/Assign_25Assignmain/q/dense/kernelsave_15/RestoreV2:25*
use_locking(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
T0
?
save_15/Assign_26Assignmain/q/dense/kernel/Adamsave_15/RestoreV2:26*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
use_locking(*
_output_shapes
:	w?*
T0
?
save_15/Assign_27Assignmain/q/dense/kernel/Adam_1save_15/RestoreV2:27*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel
?
save_15/Assign_28Assignmain/q/dense_1/biassave_15/RestoreV2:28*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0
?
save_15/Assign_29Assignmain/q/dense_1/bias/Adamsave_15/RestoreV2:29*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
T0*
_output_shapes	
:?*
use_locking(
?
save_15/Assign_30Assignmain/q/dense_1/bias/Adam_1save_15/RestoreV2:30*
validate_shape(*
T0*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
use_locking(
?
save_15/Assign_31Assignmain/q/dense_1/kernelsave_15/RestoreV2:31*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_1/kernel*
T0* 
_output_shapes
:
??
?
save_15/Assign_32Assignmain/q/dense_1/kernel/Adamsave_15/RestoreV2:32*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel
?
save_15/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_15/RestoreV2:33*
use_locking(* 
_output_shapes
:
??*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel*
T0
?
save_15/Assign_34Assignmain/q/dense_2/biassave_15/RestoreV2:34*
validate_shape(*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:
?
save_15/Assign_35Assignmain/q/dense_2/bias/Adamsave_15/RestoreV2:35*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
?
save_15/Assign_36Assignmain/q/dense_2/bias/Adam_1save_15/RestoreV2:36*
validate_shape(*
_output_shapes
:*
T0*&
_class
loc:@main/q/dense_2/bias*
use_locking(
?
save_15/Assign_37Assignmain/q/dense_2/kernelsave_15/RestoreV2:37*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(*(
_class
loc:@main/q/dense_2/kernel
?
save_15/Assign_38Assignmain/q/dense_2/kernel/Adamsave_15/RestoreV2:38*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
T0*
validate_shape(*
use_locking(
?
save_15/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_15/RestoreV2:39*
T0*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	?*
use_locking(
?
save_15/Assign_40Assigntarget/pi/dense/biassave_15/RestoreV2:40*
T0*
_output_shapes	
:?*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
use_locking(
?
save_15/Assign_41Assigntarget/pi/dense/kernelsave_15/RestoreV2:41*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?*
T0*
use_locking(*
validate_shape(
?
save_15/Assign_42Assigntarget/pi/dense_1/biassave_15/RestoreV2:42*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0*)
_class
loc:@target/pi/dense_1/bias
?
save_15/Assign_43Assigntarget/pi/dense_1/kernelsave_15/RestoreV2:43*
validate_shape(*
use_locking(* 
_output_shapes
:
??*+
_class!
loc:@target/pi/dense_1/kernel*
T0
?
save_15/Assign_44Assigntarget/pi/dense_2/biassave_15/RestoreV2:44*)
_class
loc:@target/pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
?
save_15/Assign_45Assigntarget/pi/dense_2/kernelsave_15/RestoreV2:45*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
use_locking(*
_output_shapes
:	?*
validate_shape(
?
save_15/Assign_46Assigntarget/q/dense/biassave_15/RestoreV2:46*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(*&
_class
loc:@target/q/dense/bias
?
save_15/Assign_47Assigntarget/q/dense/kernelsave_15/RestoreV2:47*
validate_shape(*(
_class
loc:@target/q/dense/kernel*
use_locking(*
T0*
_output_shapes
:	w?
?
save_15/Assign_48Assigntarget/q/dense_1/biassave_15/RestoreV2:48*
_output_shapes	
:?*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
T0*
use_locking(
?
save_15/Assign_49Assigntarget/q/dense_1/kernelsave_15/RestoreV2:49**
_class 
loc:@target/q/dense_1/kernel*
T0* 
_output_shapes
:
??*
validate_shape(*
use_locking(
?
save_15/Assign_50Assigntarget/q/dense_2/biassave_15/RestoreV2:50*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*(
_class
loc:@target/q/dense_2/bias
?
save_15/Assign_51Assigntarget/q/dense_2/kernelsave_15/RestoreV2:51*
_output_shapes
:	?*
T0*
validate_shape(**
_class 
loc:@target/q/dense_2/kernel*
use_locking(
?
save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_20^save_15/Assign_21^save_15/Assign_22^save_15/Assign_23^save_15/Assign_24^save_15/Assign_25^save_15/Assign_26^save_15/Assign_27^save_15/Assign_28^save_15/Assign_29^save_15/Assign_3^save_15/Assign_30^save_15/Assign_31^save_15/Assign_32^save_15/Assign_33^save_15/Assign_34^save_15/Assign_35^save_15/Assign_36^save_15/Assign_37^save_15/Assign_38^save_15/Assign_39^save_15/Assign_4^save_15/Assign_40^save_15/Assign_41^save_15/Assign_42^save_15/Assign_43^save_15/Assign_44^save_15/Assign_45^save_15/Assign_46^save_15/Assign_47^save_15/Assign_48^save_15/Assign_49^save_15/Assign_5^save_15/Assign_50^save_15/Assign_51^save_15/Assign_6^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard
\
save_16/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
t
save_16/filenamePlaceholderWithDefaultsave_16/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_16/ConstPlaceholderWithDefaultsave_16/filename*
shape: *
_output_shapes
: *
dtype0
?
save_16/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_33f2a4ce86964176ba44b5cb75f48fc2/part
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_16/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_16/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
?
save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
?

save_16/SaveV2/tensor_namesConst*
_output_shapes
:4*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
save_16/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2*
T0**
_class 
loc:@save_16/ShardedFilename*
_output_shapes
: 
?
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilename^save_16/control_dependency*

axis *
_output_shapes
:*
T0*
N
?
save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(
?
save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency*
_output_shapes
: *
T0
?

save_16/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
"save_16/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_16/AssignAssignbeta1_powersave_16/RestoreV2*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
?
save_16/Assign_1Assignbeta1_power_1save_16/RestoreV2:1*
use_locking(*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes
: *
validate_shape(
?
save_16/Assign_2Assignbeta2_powersave_16/RestoreV2:2*%
_class
loc:@main/pi/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
?
save_16/Assign_3Assignbeta2_power_1save_16/RestoreV2:3*$
_class
loc:@main/q/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
?
save_16/Assign_4Assignmain/pi/dense/biassave_16/RestoreV2:4*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?
?
save_16/Assign_5Assignmain/pi/dense/bias/Adamsave_16/RestoreV2:5*
use_locking(*
validate_shape(*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:?
?
save_16/Assign_6Assignmain/pi/dense/bias/Adam_1save_16/RestoreV2:6*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
?
save_16/Assign_7Assignmain/pi/dense/kernelsave_16/RestoreV2:7*
_output_shapes
:	o?*
validate_shape(*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel
?
save_16/Assign_8Assignmain/pi/dense/kernel/Adamsave_16/RestoreV2:8*
use_locking(*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
T0*
validate_shape(
?
save_16/Assign_9Assignmain/pi/dense/kernel/Adam_1save_16/RestoreV2:9*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
T0*
validate_shape(*
use_locking(
?
save_16/Assign_10Assignmain/pi/dense_1/biassave_16/RestoreV2:10*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias
?
save_16/Assign_11Assignmain/pi/dense_1/bias/Adamsave_16/RestoreV2:11*
_output_shapes	
:?*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
use_locking(
?
save_16/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_16/RestoreV2:12*'
_class
loc:@main/pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_16/Assign_13Assignmain/pi/dense_1/kernelsave_16/RestoreV2:13*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel
?
save_16/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_16/RestoreV2:14*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
T0*
validate_shape(
?
save_16/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_16/RestoreV2:15*
validate_shape(*
T0* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(
?
save_16/Assign_16Assignmain/pi/dense_2/biassave_16/RestoreV2:16*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
?
save_16/Assign_17Assignmain/pi/dense_2/bias/Adamsave_16/RestoreV2:17*
_output_shapes
:*
T0*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
validate_shape(
?
save_16/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_16/RestoreV2:18*
T0*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
use_locking(
?
save_16/Assign_19Assignmain/pi/dense_2/kernelsave_16/RestoreV2:19*
_output_shapes
:	?*
T0*)
_class
loc:@main/pi/dense_2/kernel*
use_locking(*
validate_shape(
?
save_16/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_16/RestoreV2:20*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel*
T0*
_output_shapes
:	?*
use_locking(
?
save_16/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_16/RestoreV2:21*
validate_shape(*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
save_16/Assign_22Assignmain/q/dense/biassave_16/RestoreV2:22*
_output_shapes	
:?*
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias*
use_locking(
?
save_16/Assign_23Assignmain/q/dense/bias/Adamsave_16/RestoreV2:23*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0*$
_class
loc:@main/q/dense/bias
?
save_16/Assign_24Assignmain/q/dense/bias/Adam_1save_16/RestoreV2:24*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(*$
_class
loc:@main/q/dense/bias
?
save_16/Assign_25Assignmain/q/dense/kernelsave_16/RestoreV2:25*
T0*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
use_locking(*
validate_shape(
?
save_16/Assign_26Assignmain/q/dense/kernel/Adamsave_16/RestoreV2:26*
use_locking(*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?*
validate_shape(*
T0
?
save_16/Assign_27Assignmain/q/dense/kernel/Adam_1save_16/RestoreV2:27*
validate_shape(*
T0*&
_class
loc:@main/q/dense/kernel*
use_locking(*
_output_shapes
:	w?
?
save_16/Assign_28Assignmain/q/dense_1/biassave_16/RestoreV2:28*
_output_shapes	
:?*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
T0
?
save_16/Assign_29Assignmain/q/dense_1/bias/Adamsave_16/RestoreV2:29*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
use_locking(*
T0
?
save_16/Assign_30Assignmain/q/dense_1/bias/Adam_1save_16/RestoreV2:30*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(
?
save_16/Assign_31Assignmain/q/dense_1/kernelsave_16/RestoreV2:31*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_1/kernel*
T0* 
_output_shapes
:
??
?
save_16/Assign_32Assignmain/q/dense_1/kernel/Adamsave_16/RestoreV2:32* 
_output_shapes
:
??*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel*
use_locking(*
T0
?
save_16/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_16/RestoreV2:33*
use_locking(* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
T0*
validate_shape(
?
save_16/Assign_34Assignmain/q/dense_2/biassave_16/RestoreV2:34*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
T0
?
save_16/Assign_35Assignmain/q/dense_2/bias/Adamsave_16/RestoreV2:35*
validate_shape(*
T0*&
_class
loc:@main/q/dense_2/bias*
use_locking(*
_output_shapes
:
?
save_16/Assign_36Assignmain/q/dense_2/bias/Adam_1save_16/RestoreV2:36*
T0*&
_class
loc:@main/q/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save_16/Assign_37Assignmain/q/dense_2/kernelsave_16/RestoreV2:37*
T0*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
_output_shapes
:	?
?
save_16/Assign_38Assignmain/q/dense_2/kernel/Adamsave_16/RestoreV2:38*(
_class
loc:@main/q/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(
?
save_16/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_16/RestoreV2:39*
T0*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel*
_output_shapes
:	?*
use_locking(
?
save_16/Assign_40Assigntarget/pi/dense/biassave_16/RestoreV2:40*'
_class
loc:@target/pi/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0
?
save_16/Assign_41Assigntarget/pi/dense/kernelsave_16/RestoreV2:41*
use_locking(*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	o?*
T0
?
save_16/Assign_42Assigntarget/pi/dense_1/biassave_16/RestoreV2:42*
T0*
validate_shape(*)
_class
loc:@target/pi/dense_1/bias*
use_locking(*
_output_shapes	
:?
?
save_16/Assign_43Assigntarget/pi/dense_1/kernelsave_16/RestoreV2:43* 
_output_shapes
:
??*
validate_shape(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
use_locking(
?
save_16/Assign_44Assigntarget/pi/dense_2/biassave_16/RestoreV2:44*
use_locking(*)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
?
save_16/Assign_45Assigntarget/pi/dense_2/kernelsave_16/RestoreV2:45*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	?*
use_locking(*
T0*
validate_shape(
?
save_16/Assign_46Assigntarget/q/dense/biassave_16/RestoreV2:46*
use_locking(*&
_class
loc:@target/q/dense/bias*
T0*
_output_shapes	
:?*
validate_shape(
?
save_16/Assign_47Assigntarget/q/dense/kernelsave_16/RestoreV2:47*
T0*
use_locking(*
_output_shapes
:	w?*(
_class
loc:@target/q/dense/kernel*
validate_shape(
?
save_16/Assign_48Assigntarget/q/dense_1/biassave_16/RestoreV2:48*
T0*
validate_shape(*
use_locking(*(
_class
loc:@target/q/dense_1/bias*
_output_shapes	
:?
?
save_16/Assign_49Assigntarget/q/dense_1/kernelsave_16/RestoreV2:49*
validate_shape(*
use_locking(*
T0**
_class 
loc:@target/q/dense_1/kernel* 
_output_shapes
:
??
?
save_16/Assign_50Assigntarget/q/dense_2/biassave_16/RestoreV2:50*
use_locking(*(
_class
loc:@target/q/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
?
save_16/Assign_51Assigntarget/q/dense_2/kernelsave_16/RestoreV2:51*
validate_shape(*
_output_shapes
:	?**
_class 
loc:@target/q/dense_2/kernel*
T0*
use_locking(
?
save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_16^save_16/Assign_17^save_16/Assign_18^save_16/Assign_19^save_16/Assign_2^save_16/Assign_20^save_16/Assign_21^save_16/Assign_22^save_16/Assign_23^save_16/Assign_24^save_16/Assign_25^save_16/Assign_26^save_16/Assign_27^save_16/Assign_28^save_16/Assign_29^save_16/Assign_3^save_16/Assign_30^save_16/Assign_31^save_16/Assign_32^save_16/Assign_33^save_16/Assign_34^save_16/Assign_35^save_16/Assign_36^save_16/Assign_37^save_16/Assign_38^save_16/Assign_39^save_16/Assign_4^save_16/Assign_40^save_16/Assign_41^save_16/Assign_42^save_16/Assign_43^save_16/Assign_44^save_16/Assign_45^save_16/Assign_46^save_16/Assign_47^save_16/Assign_48^save_16/Assign_49^save_16/Assign_5^save_16/Assign_50^save_16/Assign_51^save_16/Assign_6^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9
3
save_16/restore_allNoOp^save_16/restore_shard
\
save_17/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_17/filenamePlaceholderWithDefaultsave_17/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_17/ConstPlaceholderWithDefaultsave_17/filename*
shape: *
dtype0*
_output_shapes
: 
?
save_17/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_56076c9031c6434190f8b5e5fd7ba007/part*
_output_shapes
: 
~
save_17/StringJoin
StringJoinsave_17/Constsave_17/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_17/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_17/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
?
save_17/ShardedFilenameShardedFilenamesave_17/StringJoinsave_17/ShardedFilename/shardsave_17/num_shards*
_output_shapes
: 
?

save_17/SaveV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
save_17/SaveV2/shape_and_slicesConst*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_17/SaveV2SaveV2save_17/ShardedFilenamesave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_17/control_dependencyIdentitysave_17/ShardedFilename^save_17/SaveV2**
_class 
loc:@save_17/ShardedFilename*
_output_shapes
: *
T0
?
.save_17/MergeV2Checkpoints/checkpoint_prefixesPacksave_17/ShardedFilename^save_17/control_dependency*
_output_shapes
:*

axis *
N*
T0
?
save_17/MergeV2CheckpointsMergeV2Checkpoints.save_17/MergeV2Checkpoints/checkpoint_prefixessave_17/Const*
delete_old_dirs(
?
save_17/IdentityIdentitysave_17/Const^save_17/MergeV2Checkpoints^save_17/control_dependency*
T0*
_output_shapes
: 
?

save_17/RestoreV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
"save_17/RestoreV2/shape_and_slicesConst*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices*B
dtypes8
624*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save_17/AssignAssignbeta1_powersave_17/RestoreV2*
validate_shape(*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
use_locking(
?
save_17/Assign_1Assignbeta1_power_1save_17/RestoreV2:1*
T0*$
_class
loc:@main/q/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
?
save_17/Assign_2Assignbeta2_powersave_17/RestoreV2:2*
use_locking(*
T0*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
?
save_17/Assign_3Assignbeta2_power_1save_17/RestoreV2:3*
use_locking(*
validate_shape(*
T0*$
_class
loc:@main/q/dense/bias*
_output_shapes
: 
?
save_17/Assign_4Assignmain/pi/dense/biassave_17/RestoreV2:4*
T0*
validate_shape(*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
use_locking(
?
save_17/Assign_5Assignmain/pi/dense/bias/Adamsave_17/RestoreV2:5*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias
?
save_17/Assign_6Assignmain/pi/dense/bias/Adam_1save_17/RestoreV2:6*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
?
save_17/Assign_7Assignmain/pi/dense/kernelsave_17/RestoreV2:7*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	o?*
T0*
validate_shape(*
use_locking(
?
save_17/Assign_8Assignmain/pi/dense/kernel/Adamsave_17/RestoreV2:8*'
_class
loc:@main/pi/dense/kernel*
T0*
_output_shapes
:	o?*
use_locking(*
validate_shape(
?
save_17/Assign_9Assignmain/pi/dense/kernel/Adam_1save_17/RestoreV2:9*
validate_shape(*
T0*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
use_locking(
?
save_17/Assign_10Assignmain/pi/dense_1/biassave_17/RestoreV2:10*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias
?
save_17/Assign_11Assignmain/pi/dense_1/bias/Adamsave_17/RestoreV2:11*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias
?
save_17/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_17/RestoreV2:12*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
T0*
use_locking(*
validate_shape(
?
save_17/Assign_13Assignmain/pi/dense_1/kernelsave_17/RestoreV2:13* 
_output_shapes
:
??*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
use_locking(
?
save_17/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_17/RestoreV2:14*
validate_shape(* 
_output_shapes
:
??*
T0*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel
?
save_17/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_17/RestoreV2:15* 
_output_shapes
:
??*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
T0
?
save_17/Assign_16Assignmain/pi/dense_2/biassave_17/RestoreV2:16*
validate_shape(*
use_locking(*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
T0
?
save_17/Assign_17Assignmain/pi/dense_2/bias/Adamsave_17/RestoreV2:17*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias
?
save_17/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_17/RestoreV2:18*'
_class
loc:@main/pi/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
?
save_17/Assign_19Assignmain/pi/dense_2/kernelsave_17/RestoreV2:19*
_output_shapes
:	?*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
T0
?
save_17/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_17/RestoreV2:20*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	?*
use_locking(
?
save_17/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_17/RestoreV2:21*)
_class
loc:@main/pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	?*
validate_shape(
?
save_17/Assign_22Assignmain/q/dense/biassave_17/RestoreV2:22*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save_17/Assign_23Assignmain/q/dense/bias/Adamsave_17/RestoreV2:23*$
_class
loc:@main/q/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_17/Assign_24Assignmain/q/dense/bias/Adam_1save_17/RestoreV2:24*
use_locking(*
_output_shapes	
:?*
validate_shape(*$
_class
loc:@main/q/dense/bias*
T0
?
save_17/Assign_25Assignmain/q/dense/kernelsave_17/RestoreV2:25*
validate_shape(*&
_class
loc:@main/q/dense/kernel*
use_locking(*
_output_shapes
:	w?*
T0
?
save_17/Assign_26Assignmain/q/dense/kernel/Adamsave_17/RestoreV2:26*&
_class
loc:@main/q/dense/kernel*
T0*
use_locking(*
_output_shapes
:	w?*
validate_shape(
?
save_17/Assign_27Assignmain/q/dense/kernel/Adam_1save_17/RestoreV2:27*
_output_shapes
:	w?*
T0*
use_locking(*
validate_shape(*&
_class
loc:@main/q/dense/kernel
?
save_17/Assign_28Assignmain/q/dense_1/biassave_17/RestoreV2:28*&
_class
loc:@main/q/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
?
save_17/Assign_29Assignmain/q/dense_1/bias/Adamsave_17/RestoreV2:29*&
_class
loc:@main/q/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0
?
save_17/Assign_30Assignmain/q/dense_1/bias/Adam_1save_17/RestoreV2:30*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(*&
_class
loc:@main/q/dense_1/bias
?
save_17/Assign_31Assignmain/q/dense_1/kernelsave_17/RestoreV2:31* 
_output_shapes
:
??*
T0*
use_locking(*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel
?
save_17/Assign_32Assignmain/q/dense_1/kernel/Adamsave_17/RestoreV2:32*
use_locking(*
T0* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
validate_shape(
?
save_17/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_17/RestoreV2:33*(
_class
loc:@main/q/dense_1/kernel*
T0* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save_17/Assign_34Assignmain/q/dense_2/biassave_17/RestoreV2:34*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:
?
save_17/Assign_35Assignmain/q/dense_2/bias/Adamsave_17/RestoreV2:35*
validate_shape(*
use_locking(*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias*
T0
?
save_17/Assign_36Assignmain/q/dense_2/bias/Adam_1save_17/RestoreV2:36*
use_locking(*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
_output_shapes
:
?
save_17/Assign_37Assignmain/q/dense_2/kernelsave_17/RestoreV2:37*
_output_shapes
:	?*
T0*
use_locking(*
validate_shape(*(
_class
loc:@main/q/dense_2/kernel
?
save_17/Assign_38Assignmain/q/dense_2/kernel/Adamsave_17/RestoreV2:38*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel
?
save_17/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_17/RestoreV2:39*
use_locking(*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
validate_shape(*
T0
?
save_17/Assign_40Assigntarget/pi/dense/biassave_17/RestoreV2:40*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_17/Assign_41Assigntarget/pi/dense/kernelsave_17/RestoreV2:41*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	o?
?
save_17/Assign_42Assigntarget/pi/dense_1/biassave_17/RestoreV2:42*)
_class
loc:@target/pi/dense_1/bias*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(
?
save_17/Assign_43Assigntarget/pi/dense_1/kernelsave_17/RestoreV2:43*
T0*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save_17/Assign_44Assigntarget/pi/dense_2/biassave_17/RestoreV2:44*
use_locking(*)
_class
loc:@target/pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
?
save_17/Assign_45Assigntarget/pi/dense_2/kernelsave_17/RestoreV2:45*
T0*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	?*
validate_shape(
?
save_17/Assign_46Assigntarget/q/dense/biassave_17/RestoreV2:46*
T0*
use_locking(*
validate_shape(*&
_class
loc:@target/q/dense/bias*
_output_shapes	
:?
?
save_17/Assign_47Assigntarget/q/dense/kernelsave_17/RestoreV2:47*(
_class
loc:@target/q/dense/kernel*
T0*
use_locking(*
_output_shapes
:	w?*
validate_shape(
?
save_17/Assign_48Assigntarget/q/dense_1/biassave_17/RestoreV2:48*(
_class
loc:@target/q/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_17/Assign_49Assigntarget/q/dense_1/kernelsave_17/RestoreV2:49*
use_locking(* 
_output_shapes
:
??*
T0*
validate_shape(**
_class 
loc:@target/q/dense_1/kernel
?
save_17/Assign_50Assigntarget/q/dense_2/biassave_17/RestoreV2:50*
use_locking(*
validate_shape(*(
_class
loc:@target/q/dense_2/bias*
_output_shapes
:*
T0
?
save_17/Assign_51Assigntarget/q/dense_2/kernelsave_17/RestoreV2:51*
_output_shapes
:	?*
T0**
_class 
loc:@target/q/dense_2/kernel*
use_locking(*
validate_shape(
?
save_17/restore_shardNoOp^save_17/Assign^save_17/Assign_1^save_17/Assign_10^save_17/Assign_11^save_17/Assign_12^save_17/Assign_13^save_17/Assign_14^save_17/Assign_15^save_17/Assign_16^save_17/Assign_17^save_17/Assign_18^save_17/Assign_19^save_17/Assign_2^save_17/Assign_20^save_17/Assign_21^save_17/Assign_22^save_17/Assign_23^save_17/Assign_24^save_17/Assign_25^save_17/Assign_26^save_17/Assign_27^save_17/Assign_28^save_17/Assign_29^save_17/Assign_3^save_17/Assign_30^save_17/Assign_31^save_17/Assign_32^save_17/Assign_33^save_17/Assign_34^save_17/Assign_35^save_17/Assign_36^save_17/Assign_37^save_17/Assign_38^save_17/Assign_39^save_17/Assign_4^save_17/Assign_40^save_17/Assign_41^save_17/Assign_42^save_17/Assign_43^save_17/Assign_44^save_17/Assign_45^save_17/Assign_46^save_17/Assign_47^save_17/Assign_48^save_17/Assign_49^save_17/Assign_5^save_17/Assign_50^save_17/Assign_51^save_17/Assign_6^save_17/Assign_7^save_17/Assign_8^save_17/Assign_9
3
save_17/restore_allNoOp^save_17/restore_shard
\
save_18/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_18/filenamePlaceholderWithDefaultsave_18/filename/input*
_output_shapes
: *
dtype0*
shape: 
k
save_18/ConstPlaceholderWithDefaultsave_18/filename*
dtype0*
_output_shapes
: *
shape: 
?
save_18/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_74eb0c95da674e9185bd6be6951513e5/part
~
save_18/StringJoin
StringJoinsave_18/Constsave_18/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_18/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
_
save_18/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
?
save_18/ShardedFilenameShardedFilenamesave_18/StringJoinsave_18/ShardedFilename/shardsave_18/num_shards*
_output_shapes
: 
?

save_18/SaveV2/tensor_namesConst*
_output_shapes
:4*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
save_18/SaveV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:4*
dtype0
?
save_18/SaveV2SaveV2save_18/ShardedFilenamesave_18/SaveV2/tensor_namessave_18/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_18/control_dependencyIdentitysave_18/ShardedFilename^save_18/SaveV2**
_class 
loc:@save_18/ShardedFilename*
_output_shapes
: *
T0
?
.save_18/MergeV2Checkpoints/checkpoint_prefixesPacksave_18/ShardedFilename^save_18/control_dependency*
T0*
_output_shapes
:*
N*

axis 
?
save_18/MergeV2CheckpointsMergeV2Checkpoints.save_18/MergeV2Checkpoints/checkpoint_prefixessave_18/Const*
delete_old_dirs(
?
save_18/IdentityIdentitysave_18/Const^save_18/MergeV2Checkpoints^save_18/control_dependency*
T0*
_output_shapes
: 
?

save_18/RestoreV2/tensor_namesConst*
_output_shapes
:4*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel
?
"save_18/RestoreV2/shape_and_slicesConst*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_18/RestoreV2	RestoreV2save_18/Constsave_18/RestoreV2/tensor_names"save_18/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_18/AssignAssignbeta1_powersave_18/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0
?
save_18/Assign_1Assignbeta1_power_1save_18/RestoreV2:1*
T0*
_output_shapes
: *$
_class
loc:@main/q/dense/bias*
validate_shape(*
use_locking(
?
save_18/Assign_2Assignbeta2_powersave_18/RestoreV2:2*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*%
_class
loc:@main/pi/dense/bias
?
save_18/Assign_3Assignbeta2_power_1save_18/RestoreV2:3*
_output_shapes
: *$
_class
loc:@main/q/dense/bias*
T0*
use_locking(*
validate_shape(
?
save_18/Assign_4Assignmain/pi/dense/biassave_18/RestoreV2:4*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
?
save_18/Assign_5Assignmain/pi/dense/bias/Adamsave_18/RestoreV2:5*
_output_shapes	
:?*
T0*
use_locking(*%
_class
loc:@main/pi/dense/bias*
validate_shape(
?
save_18/Assign_6Assignmain/pi/dense/bias/Adam_1save_18/RestoreV2:6*%
_class
loc:@main/pi/dense/bias*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save_18/Assign_7Assignmain/pi/dense/kernelsave_18/RestoreV2:7*
use_locking(*
_output_shapes
:	o?*
validate_shape(*
T0*'
_class
loc:@main/pi/dense/kernel
?
save_18/Assign_8Assignmain/pi/dense/kernel/Adamsave_18/RestoreV2:8*
use_locking(*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	o?
?
save_18/Assign_9Assignmain/pi/dense/kernel/Adam_1save_18/RestoreV2:9*
validate_shape(*
_output_shapes
:	o?*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel
?
save_18/Assign_10Assignmain/pi/dense_1/biassave_18/RestoreV2:10*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(*'
_class
loc:@main/pi/dense_1/bias
?
save_18/Assign_11Assignmain/pi/dense_1/bias/Adamsave_18/RestoreV2:11*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias*
use_locking(*
_output_shapes	
:?*
T0
?
save_18/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_18/RestoreV2:12*
T0*
use_locking(*
_output_shapes	
:?*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(
?
save_18/Assign_13Assignmain/pi/dense_1/kernelsave_18/RestoreV2:13*
validate_shape(*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel*
T0* 
_output_shapes
:
??
?
save_18/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_18/RestoreV2:14* 
_output_shapes
:
??*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
T0*
use_locking(
?
save_18/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_18/RestoreV2:15* 
_output_shapes
:
??*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel*
T0*
validate_shape(
?
save_18/Assign_16Assignmain/pi/dense_2/biassave_18/RestoreV2:16*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias
?
save_18/Assign_17Assignmain/pi/dense_2/bias/Adamsave_18/RestoreV2:17*
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
use_locking(*
T0
?
save_18/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_18/RestoreV2:18*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
?
save_18/Assign_19Assignmain/pi/dense_2/kernelsave_18/RestoreV2:19*)
_class
loc:@main/pi/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(
?
save_18/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_18/RestoreV2:20*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
T0*
validate_shape(
?
save_18/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_18/RestoreV2:21*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	?*
use_locking(*
validate_shape(*
T0
?
save_18/Assign_22Assignmain/q/dense/biassave_18/RestoreV2:22*
use_locking(*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias*
validate_shape(*
T0
?
save_18/Assign_23Assignmain/q/dense/bias/Adamsave_18/RestoreV2:23*$
_class
loc:@main/q/dense/bias*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
save_18/Assign_24Assignmain/q/dense/bias/Adam_1save_18/RestoreV2:24*
T0*
use_locking(*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?*
validate_shape(
?
save_18/Assign_25Assignmain/q/dense/kernelsave_18/RestoreV2:25*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel
?
save_18/Assign_26Assignmain/q/dense/kernel/Adamsave_18/RestoreV2:26*
validate_shape(*
use_locking(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
T0
?
save_18/Assign_27Assignmain/q/dense/kernel/Adam_1save_18/RestoreV2:27*
use_locking(*
_output_shapes
:	w?*&
_class
loc:@main/q/dense/kernel*
validate_shape(*
T0
?
save_18/Assign_28Assignmain/q/dense_1/biassave_18/RestoreV2:28*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(
?
save_18/Assign_29Assignmain/q/dense_1/bias/Adamsave_18/RestoreV2:29*
use_locking(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
save_18/Assign_30Assignmain/q/dense_1/bias/Adam_1save_18/RestoreV2:30*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0*&
_class
loc:@main/q/dense_1/bias
?
save_18/Assign_31Assignmain/q/dense_1/kernelsave_18/RestoreV2:31*
validate_shape(*
T0* 
_output_shapes
:
??*(
_class
loc:@main/q/dense_1/kernel*
use_locking(
?
save_18/Assign_32Assignmain/q/dense_1/kernel/Adamsave_18/RestoreV2:32* 
_output_shapes
:
??*
T0*
use_locking(*
validate_shape(*(
_class
loc:@main/q/dense_1/kernel
?
save_18/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_18/RestoreV2:33* 
_output_shapes
:
??*
validate_shape(*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel
?
save_18/Assign_34Assignmain/q/dense_2/biassave_18/RestoreV2:34*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
?
save_18/Assign_35Assignmain/q/dense_2/bias/Adamsave_18/RestoreV2:35*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
?
save_18/Assign_36Assignmain/q/dense_2/bias/Adam_1save_18/RestoreV2:36*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense_2/bias*
T0*
_output_shapes
:
?
save_18/Assign_37Assignmain/q/dense_2/kernelsave_18/RestoreV2:37*
validate_shape(*
use_locking(*
_output_shapes
:	?*
T0*(
_class
loc:@main/q/dense_2/kernel
?
save_18/Assign_38Assignmain/q/dense_2/kernel/Adamsave_18/RestoreV2:38*
use_locking(*
validate_shape(*
_output_shapes
:	?*(
_class
loc:@main/q/dense_2/kernel*
T0
?
save_18/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_18/RestoreV2:39*
T0*
validate_shape(*
_output_shapes
:	?*
use_locking(*(
_class
loc:@main/q/dense_2/kernel
?
save_18/Assign_40Assigntarget/pi/dense/biassave_18/RestoreV2:40*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(
?
save_18/Assign_41Assigntarget/pi/dense/kernelsave_18/RestoreV2:41*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	o?
?
save_18/Assign_42Assigntarget/pi/dense_1/biassave_18/RestoreV2:42*
use_locking(*
_output_shapes	
:?*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(
?
save_18/Assign_43Assigntarget/pi/dense_1/kernelsave_18/RestoreV2:43*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
T0*
use_locking(
?
save_18/Assign_44Assigntarget/pi/dense_2/biassave_18/RestoreV2:44*)
_class
loc:@target/pi/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
?
save_18/Assign_45Assigntarget/pi/dense_2/kernelsave_18/RestoreV2:45*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
_output_shapes
:	?*
validate_shape(
?
save_18/Assign_46Assigntarget/q/dense/biassave_18/RestoreV2:46*&
_class
loc:@target/q/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_18/Assign_47Assigntarget/q/dense/kernelsave_18/RestoreV2:47*(
_class
loc:@target/q/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	w?*
use_locking(
?
save_18/Assign_48Assigntarget/q/dense_1/biassave_18/RestoreV2:48*(
_class
loc:@target/q/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(
?
save_18/Assign_49Assigntarget/q/dense_1/kernelsave_18/RestoreV2:49* 
_output_shapes
:
??**
_class 
loc:@target/q/dense_1/kernel*
use_locking(*
T0*
validate_shape(
?
save_18/Assign_50Assigntarget/q/dense_2/biassave_18/RestoreV2:50*
use_locking(*
T0*(
_class
loc:@target/q/dense_2/bias*
_output_shapes
:*
validate_shape(
?
save_18/Assign_51Assigntarget/q/dense_2/kernelsave_18/RestoreV2:51**
_class 
loc:@target/q/dense_2/kernel*
T0*
_output_shapes
:	?*
validate_shape(*
use_locking(
?
save_18/restore_shardNoOp^save_18/Assign^save_18/Assign_1^save_18/Assign_10^save_18/Assign_11^save_18/Assign_12^save_18/Assign_13^save_18/Assign_14^save_18/Assign_15^save_18/Assign_16^save_18/Assign_17^save_18/Assign_18^save_18/Assign_19^save_18/Assign_2^save_18/Assign_20^save_18/Assign_21^save_18/Assign_22^save_18/Assign_23^save_18/Assign_24^save_18/Assign_25^save_18/Assign_26^save_18/Assign_27^save_18/Assign_28^save_18/Assign_29^save_18/Assign_3^save_18/Assign_30^save_18/Assign_31^save_18/Assign_32^save_18/Assign_33^save_18/Assign_34^save_18/Assign_35^save_18/Assign_36^save_18/Assign_37^save_18/Assign_38^save_18/Assign_39^save_18/Assign_4^save_18/Assign_40^save_18/Assign_41^save_18/Assign_42^save_18/Assign_43^save_18/Assign_44^save_18/Assign_45^save_18/Assign_46^save_18/Assign_47^save_18/Assign_48^save_18/Assign_49^save_18/Assign_5^save_18/Assign_50^save_18/Assign_51^save_18/Assign_6^save_18/Assign_7^save_18/Assign_8^save_18/Assign_9
3
save_18/restore_allNoOp^save_18/restore_shard
\
save_19/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_19/filenamePlaceholderWithDefaultsave_19/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_19/ConstPlaceholderWithDefaultsave_19/filename*
_output_shapes
: *
dtype0*
shape: 
?
save_19/StringJoin/inputs_1Const*<
value3B1 B+_temp_870b744098834dc5949737b11cc4d883/part*
_output_shapes
: *
dtype0
~
save_19/StringJoin
StringJoinsave_19/Constsave_19/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_19/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_19/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
?
save_19/ShardedFilenameShardedFilenamesave_19/StringJoinsave_19/ShardedFilename/shardsave_19/num_shards*
_output_shapes
: 
?

save_19/SaveV2/tensor_namesConst*
_output_shapes
:4*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
dtype0
?
save_19/SaveV2/shape_and_slicesConst*
_output_shapes
:4*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_19/SaveV2SaveV2save_19/ShardedFilenamesave_19/SaveV2/tensor_namessave_19/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q/dense/biasmain/q/dense/bias/Adammain/q/dense/bias/Adam_1main/q/dense/kernelmain/q/dense/kernel/Adammain/q/dense/kernel/Adam_1main/q/dense_1/biasmain/q/dense_1/bias/Adammain/q/dense_1/bias/Adam_1main/q/dense_1/kernelmain/q/dense_1/kernel/Adammain/q/dense_1/kernel/Adam_1main/q/dense_2/biasmain/q/dense_2/bias/Adammain/q/dense_2/bias/Adam_1main/q/dense_2/kernelmain/q/dense_2/kernel/Adammain/q/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q/dense/biastarget/q/dense/kerneltarget/q/dense_1/biastarget/q/dense_1/kerneltarget/q/dense_2/biastarget/q/dense_2/kernel*B
dtypes8
624
?
save_19/control_dependencyIdentitysave_19/ShardedFilename^save_19/SaveV2**
_class 
loc:@save_19/ShardedFilename*
_output_shapes
: *
T0
?
.save_19/MergeV2Checkpoints/checkpoint_prefixesPacksave_19/ShardedFilename^save_19/control_dependency*

axis *
T0*
_output_shapes
:*
N
?
save_19/MergeV2CheckpointsMergeV2Checkpoints.save_19/MergeV2Checkpoints/checkpoint_prefixessave_19/Const*
delete_old_dirs(
?
save_19/IdentityIdentitysave_19/Const^save_19/MergeV2Checkpoints^save_19/control_dependency*
_output_shapes
: *
T0
?

save_19/RestoreV2/tensor_namesConst*
dtype0*?

value?
B?
4Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q/dense/biasBmain/q/dense/bias/AdamBmain/q/dense/bias/Adam_1Bmain/q/dense/kernelBmain/q/dense/kernel/AdamBmain/q/dense/kernel/Adam_1Bmain/q/dense_1/biasBmain/q/dense_1/bias/AdamBmain/q/dense_1/bias/Adam_1Bmain/q/dense_1/kernelBmain/q/dense_1/kernel/AdamBmain/q/dense_1/kernel/Adam_1Bmain/q/dense_2/biasBmain/q/dense_2/bias/AdamBmain/q/dense_2/bias/Adam_1Bmain/q/dense_2/kernelBmain/q/dense_2/kernel/AdamBmain/q/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q/dense/biasBtarget/q/dense/kernelBtarget/q/dense_1/biasBtarget/q/dense_1/kernelBtarget/q/dense_2/biasBtarget/q/dense_2/kernel*
_output_shapes
:4
?
"save_19/RestoreV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
?
save_19/RestoreV2	RestoreV2save_19/Constsave_19/RestoreV2/tensor_names"save_19/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
?
save_19/AssignAssignbeta1_powersave_19/RestoreV2*
validate_shape(*
T0*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
use_locking(
?
save_19/Assign_1Assignbeta1_power_1save_19/RestoreV2:1*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@main/q/dense/bias*
validate_shape(
?
save_19/Assign_2Assignbeta2_powersave_19/RestoreV2:2*
use_locking(*
_output_shapes
: *
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(
?
save_19/Assign_3Assignbeta2_power_1save_19/RestoreV2:3*
T0*
validate_shape(*$
_class
loc:@main/q/dense/bias*
_output_shapes
: *
use_locking(
?
save_19/Assign_4Assignmain/pi/dense/biassave_19/RestoreV2:4*
T0*
_output_shapes	
:?*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
use_locking(
?
save_19/Assign_5Assignmain/pi/dense/bias/Adamsave_19/RestoreV2:5*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
use_locking(*
T0*
_output_shapes	
:?
?
save_19/Assign_6Assignmain/pi/dense/bias/Adam_1save_19/RestoreV2:6*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(*%
_class
loc:@main/pi/dense/bias
?
save_19/Assign_7Assignmain/pi/dense/kernelsave_19/RestoreV2:7*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	o?*
use_locking(
?
save_19/Assign_8Assignmain/pi/dense/kernel/Adamsave_19/RestoreV2:8*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	o?*
use_locking(
?
save_19/Assign_9Assignmain/pi/dense/kernel/Adam_1save_19/RestoreV2:9*
_output_shapes
:	o?*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
use_locking(*
T0
?
save_19/Assign_10Assignmain/pi/dense_1/biassave_19/RestoreV2:10*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(*'
_class
loc:@main/pi/dense_1/bias
?
save_19/Assign_11Assignmain/pi/dense_1/bias/Adamsave_19/RestoreV2:11*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias
?
save_19/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_19/RestoreV2:12*
T0*
use_locking(*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:?*
validate_shape(
?
save_19/Assign_13Assignmain/pi/dense_1/kernelsave_19/RestoreV2:13*
T0*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save_19/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_19/RestoreV2:14* 
_output_shapes
:
??*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(*
T0
?
save_19/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_19/RestoreV2:15*
use_locking(* 
_output_shapes
:
??*
validate_shape(*
T0*)
_class
loc:@main/pi/dense_1/kernel
?
save_19/Assign_16Assignmain/pi/dense_2/biassave_19/RestoreV2:16*
validate_shape(*
T0*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
use_locking(
?
save_19/Assign_17Assignmain/pi/dense_2/bias/Adamsave_19/RestoreV2:17*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias
?
save_19/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_19/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
validate_shape(
?
save_19/Assign_19Assignmain/pi/dense_2/kernelsave_19/RestoreV2:19*
_output_shapes
:	?*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(
?
save_19/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_19/RestoreV2:20*
T0*
_output_shapes
:	?*)
_class
loc:@main/pi/dense_2/kernel*
use_locking(*
validate_shape(
?
save_19/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_19/RestoreV2:21*
T0*
_output_shapes
:	?*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(
?
save_19/Assign_22Assignmain/q/dense/biassave_19/RestoreV2:22*$
_class
loc:@main/q/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0
?
save_19/Assign_23Assignmain/q/dense/bias/Adamsave_19/RestoreV2:23*
validate_shape(*
T0*
use_locking(*$
_class
loc:@main/q/dense/bias*
_output_shapes	
:?
?
save_19/Assign_24Assignmain/q/dense/bias/Adam_1save_19/RestoreV2:24*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*$
_class
loc:@main/q/dense/bias
?
save_19/Assign_25Assignmain/q/dense/kernelsave_19/RestoreV2:25*
_output_shapes
:	w?*
validate_shape(*
T0*&
_class
loc:@main/q/dense/kernel*
use_locking(
?
save_19/Assign_26Assignmain/q/dense/kernel/Adamsave_19/RestoreV2:26*&
_class
loc:@main/q/dense/kernel*
_output_shapes
:	w?*
validate_shape(*
use_locking(*
T0
?
save_19/Assign_27Assignmain/q/dense/kernel/Adam_1save_19/RestoreV2:27*
T0*
_output_shapes
:	w?*
validate_shape(*
use_locking(*&
_class
loc:@main/q/dense/kernel
?
save_19/Assign_28Assignmain/q/dense_1/biassave_19/RestoreV2:28*
use_locking(*&
_class
loc:@main/q/dense_1/bias*
_output_shapes	
:?*
validate_shape(*
T0
?
save_19/Assign_29Assignmain/q/dense_1/bias/Adamsave_19/RestoreV2:29*
_output_shapes	
:?*
validate_shape(*&
_class
loc:@main/q/dense_1/bias*
use_locking(*
T0
?
save_19/Assign_30Assignmain/q/dense_1/bias/Adam_1save_19/RestoreV2:30*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*&
_class
loc:@main/q/dense_1/bias
?
save_19/Assign_31Assignmain/q/dense_1/kernelsave_19/RestoreV2:31*(
_class
loc:@main/q/dense_1/kernel*
T0* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save_19/Assign_32Assignmain/q/dense_1/kernel/Adamsave_19/RestoreV2:32* 
_output_shapes
:
??*
validate_shape(*
use_locking(*
T0*(
_class
loc:@main/q/dense_1/kernel
?
save_19/Assign_33Assignmain/q/dense_1/kernel/Adam_1save_19/RestoreV2:33*
use_locking(*
validate_shape(* 
_output_shapes
:
??*
T0*(
_class
loc:@main/q/dense_1/kernel
?
save_19/Assign_34Assignmain/q/dense_2/biassave_19/RestoreV2:34*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*&
_class
loc:@main/q/dense_2/bias
?
save_19/Assign_35Assignmain/q/dense_2/bias/Adamsave_19/RestoreV2:35*
T0*&
_class
loc:@main/q/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
?
save_19/Assign_36Assignmain/q/dense_2/bias/Adam_1save_19/RestoreV2:36*
validate_shape(*
_output_shapes
:*
T0*&
_class
loc:@main/q/dense_2/bias*
use_locking(
?
save_19/Assign_37Assignmain/q/dense_2/kernelsave_19/RestoreV2:37*
_output_shapes
:	?*
validate_shape(*
use_locking(*(
_class
loc:@main/q/dense_2/kernel*
T0
?
save_19/Assign_38Assignmain/q/dense_2/kernel/Adamsave_19/RestoreV2:38*
_output_shapes
:	?*
use_locking(*
validate_shape(*
T0*(
_class
loc:@main/q/dense_2/kernel
?
save_19/Assign_39Assignmain/q/dense_2/kernel/Adam_1save_19/RestoreV2:39*(
_class
loc:@main/q/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	?
?
save_19/Assign_40Assigntarget/pi/dense/biassave_19/RestoreV2:40*
use_locking(*
_output_shapes	
:?*'
_class
loc:@target/pi/dense/bias*
T0*
validate_shape(
?
save_19/Assign_41Assigntarget/pi/dense/kernelsave_19/RestoreV2:41*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	o?*
validate_shape(*
T0*
use_locking(
?
save_19/Assign_42Assigntarget/pi/dense_1/biassave_19/RestoreV2:42*)
_class
loc:@target/pi/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save_19/Assign_43Assigntarget/pi/dense_1/kernelsave_19/RestoreV2:43*
validate_shape(* 
_output_shapes
:
??*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
use_locking(
?
save_19/Assign_44Assigntarget/pi/dense_2/biassave_19/RestoreV2:44*
validate_shape(*)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
?
save_19/Assign_45Assigntarget/pi/dense_2/kernelsave_19/RestoreV2:45*
validate_shape(*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
_output_shapes
:	?
?
save_19/Assign_46Assigntarget/q/dense/biassave_19/RestoreV2:46*
validate_shape(*
use_locking(*
T0*&
_class
loc:@target/q/dense/bias*
_output_shapes	
:?
?
save_19/Assign_47Assigntarget/q/dense/kernelsave_19/RestoreV2:47*
_output_shapes
:	w?*(
_class
loc:@target/q/dense/kernel*
T0*
use_locking(*
validate_shape(
?
save_19/Assign_48Assigntarget/q/dense_1/biassave_19/RestoreV2:48*(
_class
loc:@target/q/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_19/Assign_49Assigntarget/q/dense_1/kernelsave_19/RestoreV2:49* 
_output_shapes
:
??*
validate_shape(**
_class 
loc:@target/q/dense_1/kernel*
T0*
use_locking(
?
save_19/Assign_50Assigntarget/q/dense_2/biassave_19/RestoreV2:50*
T0*
validate_shape(*(
_class
loc:@target/q/dense_2/bias*
use_locking(*
_output_shapes
:
?
save_19/Assign_51Assigntarget/q/dense_2/kernelsave_19/RestoreV2:51*
use_locking(**
_class 
loc:@target/q/dense_2/kernel*
_output_shapes
:	?*
T0*
validate_shape(
?
save_19/restore_shardNoOp^save_19/Assign^save_19/Assign_1^save_19/Assign_10^save_19/Assign_11^save_19/Assign_12^save_19/Assign_13^save_19/Assign_14^save_19/Assign_15^save_19/Assign_16^save_19/Assign_17^save_19/Assign_18^save_19/Assign_19^save_19/Assign_2^save_19/Assign_20^save_19/Assign_21^save_19/Assign_22^save_19/Assign_23^save_19/Assign_24^save_19/Assign_25^save_19/Assign_26^save_19/Assign_27^save_19/Assign_28^save_19/Assign_29^save_19/Assign_3^save_19/Assign_30^save_19/Assign_31^save_19/Assign_32^save_19/Assign_33^save_19/Assign_34^save_19/Assign_35^save_19/Assign_36^save_19/Assign_37^save_19/Assign_38^save_19/Assign_39^save_19/Assign_4^save_19/Assign_40^save_19/Assign_41^save_19/Assign_42^save_19/Assign_43^save_19/Assign_44^save_19/Assign_45^save_19/Assign_46^save_19/Assign_47^save_19/Assign_48^save_19/Assign_49^save_19/Assign_5^save_19/Assign_50^save_19/Assign_51^save_19/Assign_6^save_19/Assign_7^save_19/Assign_8^save_19/Assign_9
3
save_19/restore_allNoOp^save_19/restore_shard "?E
save_19/Const:0save_19/Identity:0save_19/restore_all (5 @F8"?8
	variables?8?8
?
main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:021main/pi/dense/kernel/Initializer/random_uniform:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08
?
main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:023main/pi/dense_1/kernel/Initializer/random_uniform:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08
?
main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08
?
main/q/dense/kernel:0main/q/dense/kernel/Assignmain/q/dense/kernel/read:020main/q/dense/kernel/Initializer/random_uniform:08
r
main/q/dense/bias:0main/q/dense/bias/Assignmain/q/dense/bias/read:02%main/q/dense/bias/Initializer/zeros:08
?
main/q/dense_1/kernel:0main/q/dense_1/kernel/Assignmain/q/dense_1/kernel/read:022main/q/dense_1/kernel/Initializer/random_uniform:08
z
main/q/dense_1/bias:0main/q/dense_1/bias/Assignmain/q/dense_1/bias/read:02'main/q/dense_1/bias/Initializer/zeros:08
?
main/q/dense_2/kernel:0main/q/dense_2/kernel/Assignmain/q/dense_2/kernel/read:022main/q/dense_2/kernel/Initializer/random_uniform:08
z
main/q/dense_2/bias:0main/q/dense_2/bias/Assignmain/q/dense_2/bias/read:02'main/q/dense_2/bias/Initializer/zeros:08
?
target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:023target/pi/dense/kernel/Initializer/random_uniform:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08
?
target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:025target/pi/dense_1/kernel/Initializer/random_uniform:08
?
target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08
?
target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08
?
target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08
?
target/q/dense/kernel:0target/q/dense/kernel/Assigntarget/q/dense/kernel/read:022target/q/dense/kernel/Initializer/random_uniform:08
z
target/q/dense/bias:0target/q/dense/bias/Assigntarget/q/dense/bias/read:02'target/q/dense/bias/Initializer/zeros:08
?
target/q/dense_1/kernel:0target/q/dense_1/kernel/Assigntarget/q/dense_1/kernel/read:024target/q/dense_1/kernel/Initializer/random_uniform:08
?
target/q/dense_1/bias:0target/q/dense_1/bias/Assigntarget/q/dense_1/bias/read:02)target/q/dense_1/bias/Initializer/zeros:08
?
target/q/dense_2/kernel:0target/q/dense_2/kernel/Assigntarget/q/dense_2/kernel/read:024target/q/dense_2/kernel/Initializer/random_uniform:08
?
target/q/dense_2/bias:0target/q/dense_2/bias/Assigntarget/q/dense_2/bias/read:02)target/q/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
?
main/pi/dense/kernel/Adam:0 main/pi/dense/kernel/Adam/Assign main/pi/dense/kernel/Adam/read:02-main/pi/dense/kernel/Adam/Initializer/zeros:0
?
main/pi/dense/kernel/Adam_1:0"main/pi/dense/kernel/Adam_1/Assign"main/pi/dense/kernel/Adam_1/read:02/main/pi/dense/kernel/Adam_1/Initializer/zeros:0
?
main/pi/dense/bias/Adam:0main/pi/dense/bias/Adam/Assignmain/pi/dense/bias/Adam/read:02+main/pi/dense/bias/Adam/Initializer/zeros:0
?
main/pi/dense/bias/Adam_1:0 main/pi/dense/bias/Adam_1/Assign main/pi/dense/bias/Adam_1/read:02-main/pi/dense/bias/Adam_1/Initializer/zeros:0
?
main/pi/dense_1/kernel/Adam:0"main/pi/dense_1/kernel/Adam/Assign"main/pi/dense_1/kernel/Adam/read:02/main/pi/dense_1/kernel/Adam/Initializer/zeros:0
?
main/pi/dense_1/kernel/Adam_1:0$main/pi/dense_1/kernel/Adam_1/Assign$main/pi/dense_1/kernel/Adam_1/read:021main/pi/dense_1/kernel/Adam_1/Initializer/zeros:0
?
main/pi/dense_1/bias/Adam:0 main/pi/dense_1/bias/Adam/Assign main/pi/dense_1/bias/Adam/read:02-main/pi/dense_1/bias/Adam/Initializer/zeros:0
?
main/pi/dense_1/bias/Adam_1:0"main/pi/dense_1/bias/Adam_1/Assign"main/pi/dense_1/bias/Adam_1/read:02/main/pi/dense_1/bias/Adam_1/Initializer/zeros:0
?
main/pi/dense_2/kernel/Adam:0"main/pi/dense_2/kernel/Adam/Assign"main/pi/dense_2/kernel/Adam/read:02/main/pi/dense_2/kernel/Adam/Initializer/zeros:0
?
main/pi/dense_2/kernel/Adam_1:0$main/pi/dense_2/kernel/Adam_1/Assign$main/pi/dense_2/kernel/Adam_1/read:021main/pi/dense_2/kernel/Adam_1/Initializer/zeros:0
?
main/pi/dense_2/bias/Adam:0 main/pi/dense_2/bias/Adam/Assign main/pi/dense_2/bias/Adam/read:02-main/pi/dense_2/bias/Adam/Initializer/zeros:0
?
main/pi/dense_2/bias/Adam_1:0"main/pi/dense_2/bias/Adam_1/Assign"main/pi/dense_2/bias/Adam_1/read:02/main/pi/dense_2/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
?
main/q/dense/kernel/Adam:0main/q/dense/kernel/Adam/Assignmain/q/dense/kernel/Adam/read:02,main/q/dense/kernel/Adam/Initializer/zeros:0
?
main/q/dense/kernel/Adam_1:0!main/q/dense/kernel/Adam_1/Assign!main/q/dense/kernel/Adam_1/read:02.main/q/dense/kernel/Adam_1/Initializer/zeros:0
?
main/q/dense/bias/Adam:0main/q/dense/bias/Adam/Assignmain/q/dense/bias/Adam/read:02*main/q/dense/bias/Adam/Initializer/zeros:0
?
main/q/dense/bias/Adam_1:0main/q/dense/bias/Adam_1/Assignmain/q/dense/bias/Adam_1/read:02,main/q/dense/bias/Adam_1/Initializer/zeros:0
?
main/q/dense_1/kernel/Adam:0!main/q/dense_1/kernel/Adam/Assign!main/q/dense_1/kernel/Adam/read:02.main/q/dense_1/kernel/Adam/Initializer/zeros:0
?
main/q/dense_1/kernel/Adam_1:0#main/q/dense_1/kernel/Adam_1/Assign#main/q/dense_1/kernel/Adam_1/read:020main/q/dense_1/kernel/Adam_1/Initializer/zeros:0
?
main/q/dense_1/bias/Adam:0main/q/dense_1/bias/Adam/Assignmain/q/dense_1/bias/Adam/read:02,main/q/dense_1/bias/Adam/Initializer/zeros:0
?
main/q/dense_1/bias/Adam_1:0!main/q/dense_1/bias/Adam_1/Assign!main/q/dense_1/bias/Adam_1/read:02.main/q/dense_1/bias/Adam_1/Initializer/zeros:0
?
main/q/dense_2/kernel/Adam:0!main/q/dense_2/kernel/Adam/Assign!main/q/dense_2/kernel/Adam/read:02.main/q/dense_2/kernel/Adam/Initializer/zeros:0
?
main/q/dense_2/kernel/Adam_1:0#main/q/dense_2/kernel/Adam_1/Assign#main/q/dense_2/kernel/Adam_1/read:020main/q/dense_2/kernel/Adam_1/Initializer/zeros:0
?
main/q/dense_2/bias/Adam:0main/q/dense_2/bias/Adam/Assignmain/q/dense_2/bias/Adam/read:02,main/q/dense_2/bias/Adam/Initializer/zeros:0
?
main/q/dense_2/bias/Adam_1:0!main/q/dense_2/bias/Adam_1/Assign!main/q/dense_2/bias/Adam_1/read:02.main/q/dense_2/bias/Adam_1/Initializer/zeros:0"
train_op

Adam
Adam_1"?
trainable_variables??
?
main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:021main/pi/dense/kernel/Initializer/random_uniform:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08
?
main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:023main/pi/dense_1/kernel/Initializer/random_uniform:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08
?
main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08
?
main/q/dense/kernel:0main/q/dense/kernel/Assignmain/q/dense/kernel/read:020main/q/dense/kernel/Initializer/random_uniform:08
r
main/q/dense/bias:0main/q/dense/bias/Assignmain/q/dense/bias/read:02%main/q/dense/bias/Initializer/zeros:08
?
main/q/dense_1/kernel:0main/q/dense_1/kernel/Assignmain/q/dense_1/kernel/read:022main/q/dense_1/kernel/Initializer/random_uniform:08
z
main/q/dense_1/bias:0main/q/dense_1/bias/Assignmain/q/dense_1/bias/read:02'main/q/dense_1/bias/Initializer/zeros:08
?
main/q/dense_2/kernel:0main/q/dense_2/kernel/Assignmain/q/dense_2/kernel/read:022main/q/dense_2/kernel/Initializer/random_uniform:08
z
main/q/dense_2/bias:0main/q/dense_2/bias/Assignmain/q/dense_2/bias/read:02'main/q/dense_2/bias/Initializer/zeros:08
?
target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:023target/pi/dense/kernel/Initializer/random_uniform:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08
?
target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:025target/pi/dense_1/kernel/Initializer/random_uniform:08
?
target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08
?
target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08
?
target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08
?
target/q/dense/kernel:0target/q/dense/kernel/Assigntarget/q/dense/kernel/read:022target/q/dense/kernel/Initializer/random_uniform:08
z
target/q/dense/bias:0target/q/dense/bias/Assigntarget/q/dense/bias/read:02'target/q/dense/bias/Initializer/zeros:08
?
target/q/dense_1/kernel:0target/q/dense_1/kernel/Assigntarget/q/dense_1/kernel/read:024target/q/dense_1/kernel/Initializer/random_uniform:08
?
target/q/dense_1/bias:0target/q/dense_1/bias/Assigntarget/q/dense_1/bias/read:02)target/q/dense_1/bias/Initializer/zeros:08
?
target/q/dense_2/kernel:0target/q/dense_2/kernel/Assigntarget/q/dense_2/kernel/read:024target/q/dense_2/kernel/Initializer/random_uniform:08
?
target/q/dense_2/bias:0target/q/dense_2/bias/Assigntarget/q/dense_2/bias/read:02)target/q/dense_2/bias/Initializer/zeros:08*?
serving_default?
+
a&
Placeholder_1:0?????????
)
x$
Placeholder:0?????????o(
q#
main/q/Squeeze:0?????????*
pi$
main/pi/mul:0?????????tensorflow/serving/predict