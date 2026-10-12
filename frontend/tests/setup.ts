import {afterEach,vi} from 'vitest';
import {cleanup} from '@testing-library/react';
afterEach(()=>{cleanup();vi.useRealTimers();vi.unstubAllGlobals();Object.defineProperty(document,'visibilityState',{value:'visible',configurable:true});});
Object.defineProperty(URL,'createObjectURL',{value:vi.fn(()=>`blob:fictional-${Math.random()}`),writable:true});
Object.defineProperty(URL,'revokeObjectURL',{value:vi.fn(),writable:true});
Object.defineProperty(HTMLMediaElement.prototype,'play',{value:vi.fn(()=>Promise.resolve()),writable:true});
