using NeuralCodecs.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Interfaces
{
    public interface IModelFactory
    {
        //bool CanCreateModel(string architecture);
        INeuralCodec CreateModel(IModelConfig config);
    }
}
