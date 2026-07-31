SOUNDLENS — RAILWAY DATA-SAFE UPDATE

WHAT WAS FIXED
- User/admin/report data now lives in a persistent data directory instead of
  being tied to the deployed GitHub code folder.
- Railway automatically uses /data when a Volume is mounted there.
- SOUNDLENS_DATA_DIR can override the location.
- Existing project-root data is copied into the persistent directory once,
  only when the destination is empty. Existing production data is never
  overwritten.
- JSON writes are atomic to reduce the risk of a partial or blank file.
- Production JSON files and report folders are excluded by .gitignore.
- The package contains code only. It contains no blank user-data files.

RAILWAY SETUP — REQUIRED ONCE
1. In Railway, open the SoundLens service.
2. Add a Volume and mount it at: /data
3. Optionally add variable: SOUNDLENS_DATA_DIR=/data
4. Replace the code files with this package.
5. Push:
   git add .
   git commit -m "Make SoundLens data persistent"
   git push origin main

DO NOT DELETE THE RAILWAY VOLUME.
A code rollback/deploy does not replace files stored in /data.

IMPORTANT
This update prevents future deployments from replacing data. It cannot recreate
users that are already absent unless an older JSON file, Railway volume backup,
or other backup still exists.
