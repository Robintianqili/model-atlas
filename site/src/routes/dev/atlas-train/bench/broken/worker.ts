// A worker that fails while loading, for the main-thread fallback check (/dev/atlas-train/?engine=broken).
throw new Error('this worker fails on purpose (main-thread fallback check)');
