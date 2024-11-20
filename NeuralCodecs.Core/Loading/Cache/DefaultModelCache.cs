using NeuralCodecs.Core.Exceptions;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace NeuralCodecs.Core.Loading.Cache
{
    /// <summary>
    /// Provides default implementation for model caching functionality.
    /// Manages the local storage and retrieval of downloaded models with validation and cleanup capabilities.
    /// </summary>
    public class DefaultModelCache : IModelCache
    {
        private readonly string _cacheRoot;
        private static readonly SemaphoreSlim _cacheLock = new(1);

        public DefaultModelCache(string? cacheRoot = null)
        {
            _cacheRoot = cacheRoot ?? GetDefaultCacheDirectory();
            Directory.CreateDirectory(_cacheRoot);
        }

        public string GetCacheDirectory() => _cacheRoot;

        public string GetDefaultCacheDirectory()
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "neural_codecs", "torch");
        }

        public async Task<string?> GetCachedPath(string modelId, string revision)
        {
            var modelDir = GetModelCacheDir(modelId, revision);
            var modelPath = Path.Combine(modelDir, "pytorch_model.bin");
            var configPath = Path.Combine(modelDir, "config.json");
            var metaPath = Path.Combine(modelDir, "cache_meta.json");

            if (File.Exists(modelPath) &&
                File.Exists(configPath) &&
                File.Exists(metaPath))
            {
                // Validate cache metadata
                if (await ValidateCacheMetadata(metaPath))
                {
                    return modelPath;
                }
            }

            return null;
        }

        public async Task<string> CacheModel(
            string modelId,
            string sourcePath,
            string revision,
            string targetFileName,
            string targetConfigFileName,
            IDictionary<string, string>? additionalMetadata = null)
        {
            var targetDir = GetModelCacheDir(modelId, revision);
            var targetPath = Path.Combine(targetDir, targetFileName);
            var targetConfigPath = Path.Combine(targetDir, targetConfigFileName);

            await _cacheLock.WaitAsync();
            try
            {
                // Verify source file exists and is accessible
                if (!File.Exists(Path.Combine(sourcePath, targetFileName)))
                {
                    throw new FileNotFoundException($"Source file not found: {sourcePath}");
                }

                if (!Directory.Exists(targetDir))
                {
                    Directory.CreateDirectory(targetDir);
                }

                // Copy files with explicit close
                await using (var source = File.OpenRead(Path.Combine(sourcePath, targetFileName)))
                await using (var target = File.OpenWrite(targetPath))
                {
                    await source.CopyToAsync(target);
                }
                await using (var source = File.OpenRead(Path.Combine(sourcePath, targetConfigFileName)))
                await using (var targetConfig = File.OpenWrite(targetConfigPath))
                {
                    await source.CopyToAsync(targetConfig);
                }
                //File.Copy(sourcePath, targetPath);
                //File.Copy(sourcePath, targetPath);
                await CreateCacheMetadata(targetDir, modelId, revision, sourcePath);

                return targetPath;
            }
            catch (Exception ex)
            {
                // Cleanup on failure
                if (File.Exists(targetPath))
                {
                    File.Delete(targetPath);
                }
                throw new CacheException($"Failed to cache model file: {targetFileName}", ex);
            }
            finally
            {
                _cacheLock.Release();
            }
        }
        public void ClearCache(string? modelId = null)
        {
            try
            {
                if (modelId == null)
                {
                    // Clear all cache
                    if (Directory.Exists(_cacheRoot))
                    {
                        Directory.Delete(_cacheRoot, recursive: true);
                        Directory.CreateDirectory(_cacheRoot);
                    }
                }
                else
                {
                    // Clear specific model
                    var modelDir = Path.Combine(_cacheRoot, modelId.Replace('/', '_'));
                    if (Directory.Exists(modelDir))
                    {
                        Directory.Delete(modelDir, recursive: true);
                    }
                }
            }
            catch (Exception ex)
            {
                throw new LoadException("Failed to clear cache", ex);
            }
        }

        private string GetModelCacheDir(string modelId, string revision)
        {
            var safeModelId = modelId.Replace('/', '_');
            return Path.Combine(_cacheRoot, safeModelId, revision);
        }

        private async Task<bool> ValidateCacheMetadata(string metaPath)
        {
            try
            {
                var meta = JsonSerializer.Deserialize<CacheMetadata>(
                    await File.ReadAllTextAsync(metaPath));

                if (meta == null) return false;

                // Check timestamp isn't too old (e.g., 30 days)
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

        private async Task CreateCacheMetadata(string cacheDir, string modelId, string revision, string sourcePath)
        {
            var fileNames = Directory.GetFiles(cacheDir);
            var meta = new CacheMetadata
            {
                ModelId = modelId,
                Revision = revision,
                Timestamp = DateTime.UtcNow,
                MaxAgeInDays = 30,
                Files = fileNames.Select(fileName => new CachedFile
                {
                    Path = fileName
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