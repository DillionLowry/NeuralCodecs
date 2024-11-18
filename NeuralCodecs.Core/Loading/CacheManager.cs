using NeuralCodecs.Core.Exceptions;
using System.Text.Json;

namespace NeuralCodecs.Core.Loading
{
    /// <summary>
    /// Manages caching of downloaded models and related files
    /// </summary>
    public class CacheManager
    {
        private readonly string _cacheRoot;
        private const string DEFAULT_CACHE_DIR = ".cache/neural_codecs";
        private static readonly SemaphoreSlim _cacheLock = new(1);

        public CacheManager(string? cacheRoot = null)
        {
            _cacheRoot = cacheRoot ?? GetDefaultCacheDirectory();
            Directory.CreateDirectory(_cacheRoot);
        }

        /// <summary>
        /// Gets the path to a cached model if it exists
        /// </summary>
        /// <returns>Path to cached model or null if not cached</returns>
        public string? GetCachedModel(string modelId, string revision = "main")
        {
            var cacheDir = GetModelCacheDir(modelId, revision);
            var modelPath = Path.Combine(cacheDir, "pytorch_model.bin");
            var configPath = Path.Combine(cacheDir, "config.json");
            var metaPath = Path.Combine(cacheDir, "cache_meta.json");

            // Check if files exist and metadata is valid
            if (File.Exists(modelPath) &&
                File.Exists(configPath) &&
                File.Exists(metaPath) &&
                ValidateCacheMetadata(metaPath))
            {
                return modelPath;
            }

            return null;
        }

        /// <summary>
        /// Downloads and caches a model
        /// </summary>
        /// <returns>Path to cached model</returns>
        public async Task<string> CacheModel(
            string modelId,
            string revision,
            HuggingFaceLoader loader,
            CancellationToken ct = default)
        {
            var cacheDir = GetModelCacheDir(modelId, revision);
            var lockFile = Path.Combine(cacheDir, "download.lock");

            await _cacheLock.WaitAsync(ct);
            try
            {
                // Create cache directory
                Directory.CreateDirectory(cacheDir);

                // Acquire file lock
                using var fileLock = new FileStream(
                    lockFile,
                    FileMode.OpenOrCreate,
                    FileAccess.ReadWrite,
                    FileShare.None);

                // Download files
                var progress = new Progress<double>(p =>
                    Console.WriteLine($"Download progress: {p:P0}"));

                var files = await loader.DownloadSnapshot(
                    modelId,
                    cacheDir,
                    allowedPatterns: new[] { "*.bin", "*.json" },
                    progress: progress,
                    ct: ct);

                // Validate downloaded files
                ValidateDownloadedFiles(cacheDir, files);

                // Create cache metadata
                await CreateCacheMetadata(cacheDir, modelId, revision, files);

                return Path.Combine(cacheDir, "pytorch_model.bin");
            }
            catch (Exception ex)
            {
                // Cleanup on failure
                try
                {
                    if (Directory.Exists(cacheDir))
                    {
                        Directory.Delete(cacheDir, recursive: true);
                    }
                }
                catch
                {
                    // Ignore cleanup errors
                }

                throw new ModelLoadException("Failed to cache model", ex);
            }
            finally
            {
                _cacheLock.Release();
            }
        }

        /// <summary>
        /// Clears the entire cache or specific model
        /// </summary>
        public void ClearCache(string? modelId = null)
        {
            if (modelId == null)
            {
                // Clear all
                try
                {
                    Directory.Delete(_cacheRoot, recursive: true);
                    Directory.CreateDirectory(_cacheRoot);
                }
                catch (Exception ex)
                {
                    throw new ModelLoadException("Failed to clear cache", ex);
                }
            }
            else
            {
                // Clear specific model
                var modelDir = Path.Combine(_cacheRoot, modelId.Replace('/', '_'));
                if (Directory.Exists(modelDir))
                {
                    try
                    {
                        Directory.Delete(modelDir, recursive: true);
                    }
                    catch (Exception ex)
                    {
                        throw new ModelLoadException($"Failed to clear cache for {modelId}", ex);
                    }
                }
            }
        }

        private string GetDefaultCacheDirectory()
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                DEFAULT_CACHE_DIR);
        }

        public string GetModelCacheDir(string modelId, string revision)
        {
            var safeModelId = modelId.Replace('/', '_');
            return Path.Combine(_cacheRoot, safeModelId, revision);
        }

        private bool ValidateCacheMetadata(string metaPath)
        {
            try
            {
                var meta = JsonSerializer.Deserialize<CacheMetadata>(
                    File.ReadAllText(metaPath));

                if (meta == null) return false;

                // Check timestamp isn't too old
                var ageInDays = (DateTime.UtcNow - meta.Timestamp).TotalDays;
                if (ageInDays > meta.MaxAgeInDays) return false;

                // Verify file hashes
                foreach (var file in meta.Files)
                {
                    var path = Path.Combine(Path.GetDirectoryName(metaPath)!, file.Path);
                    if (!File.Exists(path)) return false;

                    var hash = ComputeFileHash(path);
                    if (hash != file.Hash) return false;
                }

                return true;
            }
            catch
            {
                return false;
            }
        }

        private void ValidateDownloadedFiles(string cacheDir, List<string> files)
        {
            var modelPath = Path.Combine(cacheDir, "pytorch_model.bin");
            var configPath = Path.Combine(cacheDir, "config.json");

            if (!File.Exists(modelPath))
                throw new ModelLoadException("Model weights file not found in downloaded content");
            if (!File.Exists(configPath))
                throw new ModelLoadException("Config file not found in downloaded content");

            // Basic format validation
            try
            {
                // Verify config is valid JSON
                using var reader = File.OpenText(configPath);
                JsonDocument.Parse(reader.ReadToEnd());

                // Verify model file has minimum size
                var modelInfo = new FileInfo(modelPath);
                if (modelInfo.Length < 1000) // Arbitrary minimum size
                    throw new ModelLoadException("Model file appears to be invalid (too small)");
            }
            catch (Exception ex)
            {
                throw new ModelLoadException("Downloaded files failed validation", ex);
            }
        }

        private async Task CreateCacheMetadata(string cacheDir, string modelId, string revision, List<string> files)
        {
            var meta = new CacheMetadata
            {
                ModelId = modelId,
                Revision = revision,
                Timestamp = DateTime.UtcNow,
                MaxAgeInDays = 30, // Cache expires after 30 days
                Files = files.Select(f => new CachedFile
                {
                    Path = Path.GetFileName(f),
                    Hash = ComputeFileHash(f)
                }).ToList()
            };

            var metaPath = Path.Combine(cacheDir, "cache_meta.json");
            await File.WriteAllTextAsync(
                metaPath,
                JsonSerializer.Serialize(meta, new JsonSerializerOptions
                {
                    WriteIndented = true
                }));
        }

        private string ComputeFileHash(string path)
        {
            using var sha256 = System.Security.Cryptography.SHA256.Create();
            using var stream = File.OpenRead(path);
            var hash = sha256.ComputeHash(stream);
            return Convert.ToBase64String(hash);
        }
    }
}