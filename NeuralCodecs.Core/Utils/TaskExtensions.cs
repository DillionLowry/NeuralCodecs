namespace NeuralCodecs.Core.Utils
{
    internal static class TaskExtensions
    {
        public static async ValueTask<T> WithTimeout<T>(
            this ValueTask<T> task,
            TimeSpan timeout,
            string operationName = "operation",
            CancellationToken ct = default)
        {
            using var timeoutCts = new CancellationTokenSource(timeout);
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(timeoutCts.Token, ct);

            try
            {
                return await task.ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested)
            {
                throw new TimeoutException($"The {operationName} timed out after {timeout.TotalSeconds:F1} seconds");
            }
        }

        public static async ValueTask<T> WithRetry<T>(
            this Func<CancellationToken, ValueTask<T>> operation,
            int maxAttempts = 3,
            TimeSpan? delay = null,
            CancellationToken ct = default)
        {
            delay ??= TimeSpan.FromSeconds(1);
            Exception lastException = null;

            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                try
                {
                    return await operation(ct).ConfigureAwait(false);
                }
                catch (Exception ex) when (attempt < maxAttempts - 1 &&
                                        !(ex is OperationCanceledException))
                {
                    lastException = ex;
                    await Task.Delay(delay.Value, ct).ConfigureAwait(false);
                }
            }

            throw new AggregateException($"Operation failed after {maxAttempts} attempts", lastException);
        }
    }
}