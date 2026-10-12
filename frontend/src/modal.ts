import {useEffect,useRef} from 'react';

/** Keep keyboard focus inside the active dialog and restore its opener. */
export function useModal(active:boolean,dismiss:()=>void){
  const root=useRef<HTMLElement|null>(null),close=useRef(dismiss);close.current=dismiss;
  useEffect(()=>{
    if(!active||!root.current)return;
    const dialog=root.current,previous=document.activeElement;
    const controls=()=>Array.from(dialog.querySelectorAll<HTMLElement>('button,input,select,[tabindex]')).filter(element=>!element.hasAttribute('disabled')&&element.tabIndex>=0);
    if(!dialog.contains(document.activeElement))(dialog.querySelector<HTMLElement>('input:not(:disabled)')??controls()[0])?.focus();
    const key=(event:KeyboardEvent)=>{
      if(event.key==='Escape'){event.preventDefault();close.current();return;}
      if(event.key!=='Tab')return;
      const items=controls(),first=items[0],last=items.at(-1);
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
    };
    document.addEventListener('keydown',key);
    return()=>{document.removeEventListener('keydown',key);if(previous instanceof HTMLElement&&previous.isConnected)previous.focus();};
  },[active]);
  return root;
}
