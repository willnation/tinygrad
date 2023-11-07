import unittest
from tinygrad.helpers import Timing, CI, dedup
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, BufferOps
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.shape.view import View
from test.test_net_speed import start_profile, stop_profile

class TestWinograd(unittest.TestCase):
  def setUp(self):
    self.old = Tensor.wino
    Tensor.wino = 1
  def tearDown(self): Tensor.wino = self.old

  def test_sts(self):
    x = Tensor.empty(1,4,9,9)
    w = Tensor.empty(4,4,3,3)

    with Timing("running conv: "):
      out = Tensor.conv2d(x, w)

    with Timing("scheduling: "):
      sched = out.lazydata.schedule()

    for i,si in enumerate(sched):
      if si.ast.op in LoadOps: continue
      if i != 5: continue
      sts = []
      for op in si.ast.get_lazyops():
        if op.op in BufferOps:
          sts.append(op.arg.st)
      print(f"kernel {i} with {len(sts)} shapetrackers, {len(dedup(sts))} unique")
      v = [View.create(st.views[-1].shape, st.views[-1].strides, st.views[-1].offset * 0, st.views[-1].mask) for st in sts]
      for i,st in enumerate(sorted(dedup(v))):
        print(f"{i:3d}", st)

  def test_speed(self):
    x = Tensor.empty(1,4,9,9)
    w = Tensor.empty(4,4,3,3)

    with Timing("running conv: "):
      out = Tensor.conv2d(x, w)

    with Timing("scheduling: "):
      sched = out.lazydata.schedule()

    for i,s in enumerate(sched):
      if s.ast.op in LoadOps: continue
      ops = s.ast.get_lazyops()
      with Timing(f"linearize {i} with {len(ops):4d} ops: "):
        l = Linearizer(s.ast)
        l.hand_coded_optimizations()
        l.linearize()

  def test_profile(self):
    x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
    if not CI: pr = start_profile()
    out = Tensor.conv2d(x,w).realize()
    if not CI: stop_profile(pr, sort='time')
    out.numpy()

if __name__ == '__main__':
  unittest.main(verbosity=2)