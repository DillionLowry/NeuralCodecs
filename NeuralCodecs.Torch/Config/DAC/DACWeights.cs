using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace NeuralCodecs.Torch.Config.DAC
{
    public class DACWeights
    {
        public Dictionary<string, Tensor> StateDict { get; set; }
        public Dictionary<string, object> Metadata { get; set; }
    }
}
