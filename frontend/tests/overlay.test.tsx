import React from 'react';
import {it,expect} from 'vitest';
import {render,screen} from '@testing-library/react';
import {FaceOverlay} from '../src/FaceOverlay';
import type {Analysis} from '../src/types';

function result(y=70):Analysis{return {face_count:1,face_detected:true,verified:false,threshold:null,faces:[{face_index:0,identity_diagnostic:{identity_status:'Unknown',candidate_count:0,possible_candidates:[],verified:false,threshold:null,reason:'not requested'},face_box:{x:25,y,width:150,height:120,image_width:200,image_height:200},visible_expression_prediction:{label:'neutral',softmax_score:0.2,score_calibrated:false,not_a_mood_or_diagnosis:true,domain_accuracy_validated:false}}]};}
it('anchors Emotion caption above original-frame box and labels uncertainty',()=>{render(<FaceOverlay result={result()}/>);const svg=screen.getByRole('img',{name:'Opt-in face outline'});expect(svg.getAttribute('viewBox')).toBe('0 0 200 200');expect(screen.getByText('Emotion: neutral')).toBeTruthy();expect(screen.getByText('Uncalibrated prediction')).toBeTruthy();expect(svg.querySelector('g')?.getAttribute('transform')).toBe('translate(20 36)');expect(svg.querySelector('rect')?.getAttribute('y')).toBe('70');});
it('clamps the caption at the image top and removes it when no face remains',()=>{const {rerender}=render(<FaceOverlay result={result(10)}/>);expect(screen.getByRole('img').querySelector('g')?.getAttribute('transform')).toBe('translate(20 0)');rerender(<FaceOverlay result={{...result(),face_count:0,faces:[]}}/>);expect(screen.queryByRole('img')).toBeNull();expect(screen.queryByText('Emotion: neutral')).toBeNull();});
