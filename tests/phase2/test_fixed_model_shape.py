import hashlib
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import pytest
from face_verification.gpu import fixed_model_bytes
from face_verification import VerificationError, Status

def test_static_input_keeps_original_weights_and_file(tmp_path):
    weights = np.arange(12,dtype=np.float32).reshape(4,3,1,1)
    graph = helper.make_graph([helper.make_node('Conv',['x','weights'],['y'])],'dynamic',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,['batch',3,'height','width'])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,['batch',4,'height','width'])],
        [numpy_helper.from_array(weights,'weights')])
    path = tmp_path/'original.onnx'
    onnx.save(helper.make_model(graph),path)
    original = path.read_bytes()
    fixed = onnx.load_model_from_string(fixed_model_bytes(path,(1,3,32,32)))
    assert path.read_bytes() == original
    assert [d.dim_value for d in fixed.graph.input[0].type.tensor_type.shape.dim] == [1,3,32,32]
    np.testing.assert_array_equal(numpy_helper.to_array(fixed.graph.initializer[0]),weights)
    assert list(fixed.graph.node) == list(graph.node)

def test_static_contract_mismatch_fails(tmp_path):
    graph = helper.make_graph([helper.make_node('Identity',['x'],['y'])],'fixed',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,3,16,16])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,3,16,16])])
    path = tmp_path/'fixed.onnx'
    onnx.save(helper.make_model(graph),path)
    with pytest.raises(VerificationError) as exc:
        fixed_model_bytes(path,(1,3,32,32))
    assert exc.value.status == Status.MODEL_ERROR
