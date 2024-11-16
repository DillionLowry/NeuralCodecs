using NeuralCodecs.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Exceptions
{
    public class ModelConfigException : Exception
    {
        public ModelConfigException(string message) : base(message) { }
        public ModelConfigException(string message, Exception inner) : base(message, inner) { }
    }
}