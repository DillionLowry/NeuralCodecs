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
        public Dictionary<string, Tensor> StateDict { get; }
        public Dictionary<string, object> Metadata { get; }

        public DACWeights(
            Dictionary<string, Tensor> stateDict,
            Dictionary<string, object> metadata)
        {
            StateDict = stateDict ?? throw new ArgumentNullException(nameof(stateDict));
            Metadata = metadata ?? throw new ArgumentNullException(nameof(metadata));
        }
    }
}
