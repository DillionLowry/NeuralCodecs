namespace NeuralCodecs.Core.Utils
{
    public enum AudioInterleaving
    {
        /// <summary>
        /// Audio samples are interleaved (LRLRLR...)
        /// </summary>
        Interleaved,
        
        /// <summary>
        /// Audio samples are deinterleaved (LLL...RRR...)
        /// </summary>
        Deinterleaved
        
    }
}
