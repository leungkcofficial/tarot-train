import React, { createContext, useContext, useReducer, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

/**
 * Session management context for TAROT CKD Risk Prediction
 * Handles session state, form data, and predictions without persistence
 */

// Types
export interface Demographics {
  age?: number;
  dateOfBirth?: Date;
  gender: 'male' | 'female' | '';
  ckdDiagnosisDate?: Date;
}

export interface LabValue {
  id: string;
  parameter: string;
  value: number | '';
  unit: string;
  date: Date;
}

export interface Comorbidity {
  id: string;
  condition: string;
  diagnosed: boolean;
  diagnosisDate?: Date;
}

export interface SessionData {
  demographics: Demographics;
  labValues: LabValue[];
  comorbidities: Comorbidity[];
  currentStep: number;
  sessionId: string;
  startedAt: Date;
  lastModified: Date;
}

export interface PredictionResult {
  predictions: {
    dialysis_risk: number[];
    mortality_risk: number[];
  };
  confidence_intervals?: {
    dialysis_lower: number[];
    dialysis_upper: number[];
    mortality_lower: number[];
    mortality_upper: number[];
  };
  shap_values?: {
    dialysis: Record<string, number>;
    mortality: Record<string, number>;
  };
  patient_context: {
    age: number;
    gender: string;
    egfr: number;
    egfr_stage: string;
    cci_score: number;
    observation_period: number;
  };
  clinical_benchmarks: {
    nephrology_referral_threshold: number;
    multidisciplinary_care_threshold: number;
    krt_preparation_threshold: number;
  };
  model_info: {
    ensemble_size: number;
    model_types: Record<string, number>;
    inference_time_ms: number;
    preprocessing_time_ms: number;
    sequence_length: number;
  };
  session_id: string;
  timestamp: string;
}

interface SessionState {
  data: SessionData;
  prediction?: PredictionResult;
  loading: boolean;
  error?: string;
}

// Action types
type SessionAction =
  | { type: 'UPDATE_DEMOGRAPHICS'; payload: Partial<Demographics> }
  | { type: 'ADD_LAB_VALUE'; payload: LabValue }
  | { type: 'UPDATE_LAB_VALUE'; payload: { id: string; updates: Partial<LabValue> } }
  | { type: 'REMOVE_LAB_VALUE'; payload: string }
  | { type: 'ADD_COMORBIDITY'; payload: Comorbidity }
  | { type: 'UPDATE_COMORBIDITY'; payload: { id: string; updates: Partial<Comorbidity> } }
  | { type: 'REMOVE_COMORBIDITY'; payload: string }
  | { type: 'SET_STEP'; payload: number }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | undefined }
  | { type: 'SET_PREDICTION'; payload: PredictionResult }
  | { type: 'CLEAR_SESSION' }
  | { type: 'TOUCH_SESSION' };

// Initial state
const createInitialState = (): SessionState => ({
  data: {
    demographics: {
      age: undefined,
      dateOfBirth: undefined,
      gender: '',
    },
    labValues: [],
    comorbidities: [],
    currentStep: 0,
    sessionId: uuidv4(),
    startedAt: new Date(),
    lastModified: new Date(),
  },
  loading: false,
});

// Reducer
const sessionReducer = (state: SessionState, action: SessionAction): SessionState => {
  const now = new Date();
  
  switch (action.type) {
    case 'UPDATE_DEMOGRAPHICS':
      return {
        ...state,
        data: {
          ...state.data,
          demographics: {
            ...state.data.demographics,
            ...action.payload,
          },
          lastModified: now,
        },
      };
      
    case 'ADD_LAB_VALUE':
      return {
        ...state,
        data: {
          ...state.data,
          labValues: [...state.data.labValues, action.payload],
          lastModified: now,
        },
      };
      
    case 'UPDATE_LAB_VALUE':
      return {
        ...state,
        data: {
          ...state.data,
          labValues: state.data.labValues.map(lab =>
            lab.id === action.payload.id
              ? { ...lab, ...action.payload.updates }
              : lab
          ),
          lastModified: now,
        },
      };
      
    case 'REMOVE_LAB_VALUE':
      return {
        ...state,
        data: {
          ...state.data,
          labValues: state.data.labValues.filter(lab => lab.id !== action.payload),
          lastModified: now,
        },
      };
      
    case 'ADD_COMORBIDITY':
      return {
        ...state,
        data: {
          ...state.data,
          comorbidities: [...state.data.comorbidities, action.payload],
          lastModified: now,
        },
      };
      
    case 'UPDATE_COMORBIDITY':
      return {
        ...state,
        data: {
          ...state.data,
          comorbidities: state.data.comorbidities.map(comorb =>
            comorb.id === action.payload.id
              ? { ...comorb, ...action.payload.updates }
              : comorb
          ),
          lastModified: now,
        },
      };
      
    case 'REMOVE_COMORBIDITY':
      return {
        ...state,
        data: {
          ...state.data,
          comorbidities: state.data.comorbidities.filter(comorb => comorb.id !== action.payload),
          lastModified: now,
        },
      };
      
    case 'SET_STEP':
      return {
        ...state,
        data: {
          ...state.data,
          currentStep: action.payload,
          lastModified: now,
        },
      };
      
    case 'SET_LOADING':
      return {
        ...state,
        loading: action.payload,
      };
      
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        loading: false,
      };
      
    case 'SET_PREDICTION':
      return {
        ...state,
        prediction: action.payload,
        loading: false,
        error: undefined,
      };
      
    case 'CLEAR_SESSION':
      return createInitialState();
      
    case 'TOUCH_SESSION':
      return {
        ...state,
        data: {
          ...state.data,
          lastModified: now,
        },
      };
      
    default:
      return state;
  }
};

// Context
interface SessionContextValue {
  state: SessionState;
  updateDemographics: (updates: Partial<Demographics>) => void;
  addLabValue: (labValue: Omit<LabValue, 'id'>) => void;
  updateLabValue: (id: string, updates: Partial<LabValue>) => void;
  removeLabValue: (id: string) => void;
  addComorbidity: (comorbidity: Omit<Comorbidity, 'id'>) => void;
  updateComorbidity: (id: string, updates: Partial<Comorbidity>) => void;
  removeComorbidity: (id: string) => void;
  setStep: (step: number) => void;
  nextStep: () => void;
  prevStep: () => void;
  clearSession: () => void;
  setPrediction: (prediction: PredictionResult) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | undefined) => void;
}

const SessionContext = createContext<SessionContextValue | undefined>(undefined);

// Provider component
export const SessionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(sessionReducer, undefined, createInitialState);

  // Actions
  const updateDemographics = useCallback((updates: Partial<Demographics>) => {
    dispatch({ type: 'UPDATE_DEMOGRAPHICS', payload: updates });
  }, []);

  const addLabValue = useCallback((labValue: Omit<LabValue, 'id'>) => {
    dispatch({ type: 'ADD_LAB_VALUE', payload: { ...labValue, id: uuidv4() } });
  }, []);

  const updateLabValue = useCallback((id: string, updates: Partial<LabValue>) => {
    dispatch({ type: 'UPDATE_LAB_VALUE', payload: { id, updates } });
  }, []);

  const removeLabValue = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_LAB_VALUE', payload: id });
  }, []);

  const addComorbidity = useCallback((comorbidity: Omit<Comorbidity, 'id'>) => {
    dispatch({ type: 'ADD_COMORBIDITY', payload: { ...comorbidity, id: uuidv4() } });
  }, []);

  const updateComorbidity = useCallback((id: string, updates: Partial<Comorbidity>) => {
    dispatch({ type: 'UPDATE_COMORBIDITY', payload: { id, updates } });
  }, []);

  const removeComorbidity = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_COMORBIDITY', payload: id });
  }, []);

  const setStep = useCallback((step: number) => {
    dispatch({ type: 'SET_STEP', payload: step });
  }, []);

  const nextStep = useCallback(() => {
    dispatch({ type: 'SET_STEP', payload: state.data.currentStep + 1 });
  }, [state.data.currentStep]);

  const prevStep = useCallback(() => {
    dispatch({ type: 'SET_STEP', payload: Math.max(0, state.data.currentStep - 1) });
  }, [state.data.currentStep]);

  const clearSession = useCallback(() => {
    dispatch({ type: 'CLEAR_SESSION' });
  }, []);

  const setPrediction = useCallback((prediction: PredictionResult) => {
    dispatch({ type: 'SET_PREDICTION', payload: prediction });
  }, []);

  const setLoading = useCallback((loading: boolean) => {
    dispatch({ type: 'SET_LOADING', payload: loading });
  }, []);

  const setError = useCallback((error: string | undefined) => {
    dispatch({ type: 'SET_ERROR', payload: error });
  }, []);

  // Session timeout (clear after 4 hours of inactivity)
  useEffect(() => {
    const timeout = setTimeout(() => {
      const timeSinceLastModified = Date.now() - state.data.lastModified.getTime();
      const fourHours = 4 * 60 * 60 * 1000;
      
      if (timeSinceLastModified > fourHours) {
        console.log('Session expired due to inactivity');
        clearSession();
      }
    }, 5 * 60 * 1000); // Check every 5 minutes

    return () => clearTimeout(timeout);
  }, [state.data.lastModified, clearSession]);

  // Privacy reminder in console
  useEffect(() => {
    console.log('🔒 Session initialized - No data persistence');
  }, []);

  const value: SessionContextValue = {
    state,
    updateDemographics,
    addLabValue,
    updateLabValue,
    removeLabValue,
    addComorbidity,
    updateComorbidity,
    removeComorbidity,
    setStep,
    nextStep,
    prevStep,
    clearSession,
    setPrediction,
    setLoading,
    setError,
  };

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  );
};

// Hook
export const useSession = () => {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
};