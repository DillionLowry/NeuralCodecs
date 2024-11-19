using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Models
{
    public record ModelOperationResult<T>
    {
        public bool Success { get; init; }
        public T Result { get; init; }
        public Exception Error { get; init; }
        public string Message { get; init; }

        public static ModelOperationResult<T> FromSuccess(T result) =>
            new() { Success = true, Result = result };

        public static ModelOperationResult<T> FromError(Exception ex) =>
            new() { Success = false, Error = ex, Message = $"{ex.Message} {ex.InnerException?.Message}" };
    }
}
