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
    /// <summary>
    /// Exception thrown when model loading fails
    /// </summary>
    public class ModelLoadException : Exception
    {
        public ModelLoadException(string message) : base(message) { }
        public ModelLoadException(string message, Exception inner) : base(message, inner) { }
    }
}