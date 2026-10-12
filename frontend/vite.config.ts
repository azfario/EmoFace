import {defineConfig} from 'vitest/config';
export default defineConfig({
  base:'/app/',
  build:{outDir:'dist',sourcemap:false},
  test:{environment:'jsdom',setupFiles:['./tests/setup.ts'],restoreMocks:true,fileParallelism:false,maxWorkers:1,
    reporters:['default','json'],outputFile:{json:'../reports/phase5/frontend_tests.json'}}
});
