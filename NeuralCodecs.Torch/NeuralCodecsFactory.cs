using NeuralCodecs.Torch.Loading;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralCodecs.Torch
{
    public static partial class NeuralCodecsFactory
    {
        public static NeuralCodecs CreateTorchInstance()
        {
            return new NeuralCodecs(new TorchModelLoader());
        }
    }
}
