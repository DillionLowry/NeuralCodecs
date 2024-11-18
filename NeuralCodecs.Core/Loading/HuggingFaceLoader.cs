using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Models;
using System.Text.Json;

namespace NeuralCodecs.Core.Loading;

/// <summary>
/// Handles downloading models and files from the Hugging Face Hub
/// </summary>
public class HuggingFaceLoader
{
    private readonly HttpClient _client;
    private const string API_BASE = "https://huggingface.co/api";
    private const int DEFAULT_TIMEOUT_SECONDS = 300;

    public HuggingFaceLoader(string? authToken = null)
    {
        _client = CreateHttpClient(authToken);
    }

    /// <summary>
    /// Downloads repository snapshot with specified file patterns
    /// </summary>
    public async Task<List<string>> DownloadSnapshot(
        string repoId,
        string targetDir,
        string[]? allowedPatterns = null,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        var downloadedFiles = new List<string>();
        var repoFiles = await GetRepositoryFiles(repoId, ct);

        // Filter files based on patterns
        var filesToDownload = FilterFiles(repoFiles, allowedPatterns);

        // Calculate total size for progress
        var totalSize = filesToDownload.Sum(f => f.Size);
        var downloadedSize = 0L;
        var progressHandler = new Progress<long>(bytesDownloaded =>
        {
            downloadedSize += bytesDownloaded;
            progress?.Report((double)downloadedSize / totalSize);
        });

        // Download files in parallel with rate limiting
        using var semaphore = new SemaphoreSlim(3); // Max 3 concurrent downloads
        var downloadTasks = filesToDownload.Select(async file =>
        {
            await semaphore.WaitAsync(ct);
            try
            {
                var destPath = Path.Combine(targetDir, file.Path);
                await DownloadFile(repoId, file, destPath, progressHandler, ct);
                downloadedFiles.Add(destPath);
            }
            finally
            {
                semaphore.Release();
            }
        });

        await Task.WhenAll(downloadTasks);
        return downloadedFiles;
    }

    private HttpClient CreateHttpClient(string? authToken)
    {
        var client = new HttpClient
        {
            BaseAddress = new Uri("https://huggingface.co/"),
            Timeout = TimeSpan.FromSeconds(DEFAULT_TIMEOUT_SECONDS)
        };

        client.DefaultRequestHeaders.Add("User-Agent", "NeuralCodecs-Sharp/1.0");

        if (!string.IsNullOrEmpty(authToken))
        {
            client.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", authToken);
        }

        return client;
    }

    private async Task<List<RepoFile>> GetRepositoryFiles(string repoId, CancellationToken ct)
    {
        try
        {
            var response = await _client.GetAsync($"{API_BASE}/models/{repoId}/tree/main", ct);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync(ct);
            var files = JsonSerializer.Deserialize<List<RepoFile>>(json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            return files ?? throw new ModelLoadException("Failed to get repository file listing");
        }
        catch (Exception ex)
        {
            throw new ModelLoadException($"Failed to get file listing for {repoId}", ex);
        }
    }

    private List<RepoFile> FilterFiles(List<RepoFile> files, string[]? patterns)
    {
        if (patterns == null || patterns.Length == 0)
            return files;

        var wildcards = patterns.Select(p => new WildcardPattern(p)).ToList();
        return files.Where(f =>
            wildcards.Any(w => w.IsMatch(f.Path)) &&
            f.Type == "file").ToList();
    }

    private async Task DownloadFile(
        string repoId,
        RepoFile file,
        string destPath,
        IProgress<long> progress,
        CancellationToken ct)
    {
        var dir = Path.GetDirectoryName(destPath);
        if (dir != null) Directory.CreateDirectory(dir);

        var url = $"{repoId}/resolve/main/{file.Path}";
        var tempPath = destPath + ".tmp";

        try
        {
            using var response = await _client.GetAsync(
                url,
                HttpCompletionOption.ResponseHeadersRead,
                ct);
            response.EnsureSuccessStatusCode();

            using var contentStream = await response.Content.ReadAsStreamAsync(ct);
            using var fileStream = new FileStream(
                tempPath,
                FileMode.Create,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 81920,
                useAsync: true);

            var buffer = new byte[81920];
            int bytesRead;
            while ((bytesRead = await contentStream.ReadAsync(buffer, ct)) != 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), ct);
                progress.Report(bytesRead);
            }

            // Verify file size if known
            if (file.Size > 0 && fileStream.Length != file.Size)
            {
                throw new ModelLoadException(
                    $"Downloaded file size mismatch for {file.Path}. " +
                    $"Expected {file.Size} bytes but got {fileStream.Length}");
            }
        }
        catch (Exception ex)
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);

            throw new ModelLoadException($"Failed to download {file.Path}", ex);
        }

        try
        {
            // Atomic move
            if (File.Exists(destPath))
                File.Delete(destPath);
            File.Move(tempPath, destPath);
        }
        catch (Exception ex)
        {
            throw new ModelLoadException($"Failed to move downloaded file {file.Path}", ex);
        }
    }

    /// <summary>
    /// Gets repository metadata from Hugging Face
    /// </summary>
    public async Task<RepoMetadata> GetRepositoryMetadata(
        string repoId,
        CancellationToken ct = default)
    {
        try
        {
            var response = await _client.GetAsync($"{API_BASE}/models/{repoId}", ct);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync(ct);
            var metadata = JsonSerializer.Deserialize<RepoMetadata>(json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            return metadata ?? throw new ModelLoadException("Failed to get repository metadata");
        }
        catch (Exception ex)
        {
            throw new ModelLoadException($"Failed to get metadata for {repoId}", ex);
        }
    }
}