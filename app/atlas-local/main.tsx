import React from "react";
import { createRoot } from "react-dom/client";
import PrivateResults from "../components/private-results";
import PaperAtlas from "../components/paper-atlas";
import "./base.css";
createRoot(document.getElementById("root")!).render(location.pathname.replace(/\/$/, "")==="/private" ? <PrivateResults/> : <PaperAtlas hosted privateLink/>);
