/** A stale tab can refer to a chunk replaced by a deployment. Remounting React
 * does not retry a rejected lazy import, so that failure needs a page reload. */
export function needsReaderReload(error:unknown){
  const message=error instanceof Error?error.message:'';
  return /dynamically imported module|importing a module script failed|loading chunk|preload.*(?:css|module)/i.test(message);
}
