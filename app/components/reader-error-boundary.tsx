"use client";
import {Component, type ReactNode} from 'react';
import {needsReaderReload} from '@/lib/reader-errors';
/** A bad document must not take down collection navigation. */
export default class ReaderErrorBoundary extends Component<{children:ReactNode;onBack:()=>void},{failed:boolean;attempt:number;reload:boolean}> {
 state={failed:false,attempt:0,reload:false};
 static getDerivedStateFromError(error:unknown){return {failed:true,reload:needsReaderReload(error)};}
 render(){
  if(this.state.failed)return <div className="atlas-empty" role="alert"><h3>This paper couldn’t be displayed.</h3><p>You can retry the reader or return to the collection.</p><button className="atlas-primary" onClick={()=>this.state.reload?window.location.reload():this.setState(s=>({failed:false,attempt:s.attempt+1}))}>{this.state.reload?"Reload page":"Retry reader"}</button><button onClick={this.props.onBack}>Back to collection</button></div>;
  return <div key={this.state.attempt}>{this.props.children}</div>;
 }
}
