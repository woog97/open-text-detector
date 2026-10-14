import React, {lazy,Suspense,useEffect,useState} from "react";
import { createRoot } from "react-dom/client";
import PaperBrowser from "../components/paper-browser";
import "./base.css";
import {BETA} from "../lib/ui-mode";
import SiteFeedback from "../components/site-feedback";
const ModelCalibration=lazy(()=>import('../components/model-calibration'));
const SitePages=lazy(()=>import('../components/site-pages'));
// The Classifications view was retired on Oct 8, 2026 (Discover shows calibrated flagged shares); old links open the submissions list.
{const params=new URLSearchParams(location.search);if(params.get("view")==="classifications"){params.delete("view");params.delete("model");params.delete("threshold");if(!params.has("collection"))params.set("collection","iclr/2027");history.replaceState(null,"",`/?${params}`);}}
/** Parameters that belong to the submissions list and the reader; any of them means "show the list", so old shared links keep working. */
const BROWSE_KEYS=["paper","q","collection","saved","area","tag","sort","ai","mode","page","cut"];
type Route="home"|"transparency"|"architecture"|"data"|"results"|"limitations"|"manifesto"|"about"|"calibration"|"browse";
function route():Route{const p=new URLSearchParams(location.search),v=p.get("view");if(!BETA)return v==="calibration"?"calibration":"browse";if(v==="calibration"||v==="transparency"||v==="architecture"||v==="data"||v==="results"||v==="limitations"||v==="manifesto"||v==="about")return v;return BROWSE_KEYS.some(k=>p.has(k))?"browse":"home";}
function App(){
 const [r,setR]=useState<Route>(route);
 useEffect(()=>{const on=()=>setR(route());window.addEventListener("popstate",on);return()=>window.removeEventListener("popstate",on);},[]);
 if(r==="browse")return <PaperBrowser/>;
 if(r==="calibration")return <Suspense fallback={<p>Opening calibration…</p>}><ModelCalibration/></Suspense>;
 return <Suspense fallback={<p style={{padding:24}}>Opening…</p>}><SitePages page={r}/></Suspense>;
}
createRoot(document.getElementById("root")!).render(<><App/><SiteFeedback/></>);
