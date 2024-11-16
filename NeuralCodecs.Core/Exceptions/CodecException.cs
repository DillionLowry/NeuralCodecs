using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralCodecs.Core.Models;

namespace NeuralCodecs.Core.Exceptions
{
    public class CodecException : Exception
    {
        public string CodecName { get; }
        public CodecOperation Operation { get; }

        public CodecException(string message, string codecName, CodecOperation operation, Exception innerException = null)
            : base(message, innerException)
        {
            CodecName = codecName;
            Operation = operation;
        }
    }
}