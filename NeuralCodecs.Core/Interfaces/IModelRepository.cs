using NeuralCodecs.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Interfaces
{
    public interface IModelRepository
    {
        Task<string> GetModelPath(string modelId, string revision);
        Task<ModelInfo> GetModelInfo(string modelId);
        Task DownloadModel(string modelId, string targetPath, IProgress<double> progress);
    }
}
