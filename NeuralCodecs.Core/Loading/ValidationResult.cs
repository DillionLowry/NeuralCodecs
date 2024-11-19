using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Loading
{
    public record ValidationResult
    {
        public bool IsValid => Errors.Count == 0;
        public List<string> Errors { get; init; } = new();

        public static ValidationResult Success() => new();
        public static ValidationResult Failed(params string[] errors) =>
            new() { Errors = errors.ToList() };
    }
}
