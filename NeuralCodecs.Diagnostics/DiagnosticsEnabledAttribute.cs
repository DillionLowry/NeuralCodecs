// NeuralCodecs.Diagnostics/CodecDiagnostics.cs
namespace NeuralCodecs.Diagnostics
{
    /// <summary>
    /// Attribute to enable diagnostics on a module
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class DiagnosticsEnabledAttribute : Attribute
    {
        public string ModuleName { get; }

        public DiagnosticsEnabledAttribute(string moduleName = null)
        {
            ModuleName = moduleName;
        }
    }
}