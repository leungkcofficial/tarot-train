/**
 * API service for TAROT CKD Risk Prediction
 * Handles all communication with the FastAPI backend
 */

import axios, { AxiosResponse, AxiosError } from 'axios';
import { PredictionResult } from '../contexts/SessionContext';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api/v1';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    if (process.env.NODE_ENV === 'development') {
      console.log('API Request:', config.method?.toUpperCase(), config.url);
    }
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    if (process.env.NODE_ENV === 'development') {
      console.log('API Response:', response.status, response.config.url);
    }
    return response;
  },
  (error: AxiosError) => {
    console.error('API Response Error:', error.response?.status, error.response?.data);
    
    // Handle specific error cases
    if (error.response?.status === 503) {
      throw new Error('Service temporarily unavailable. Models may be loading.');
    } else if (error.response?.status === 400) {
      const errorData = error.response.data as any;
      // Handle nested error structure
      const errorMessage = errorData.message?.message || errorData.message || 'Invalid request data';
      throw new Error(errorMessage);
    } else if (error.response?.status === 500) {
      throw new Error('Server error. Please try again later.');
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please check your connection and try again.');
    } else if (!error.response) {
      throw new Error('Network error. Please check your connection.');
    }
    
    return Promise.reject(error);
  }
);

// Types for API requests
export interface DemographicsRequest {
  age?: number;
  date_of_birth?: string;
  gender: 'male' | 'female';
}

export interface LabValueRequest {
  parameter: string;
  value: number;
  unit: string;
  date: string;
}

export interface ComorbidityRequest {
  condition: string;
  diagnosed: boolean;
  date?: string;
}

export interface PredictionRequest {
  demographics: DemographicsRequest;
  laboratory_values: LabValueRequest[];
  medical_history?: ComorbidityRequest[];
  session_id?: string;
}

export interface ValidationResponse {
  valid: boolean;
  warnings: Array<{
    field: string;
    message: string;
    severity: string;
  }>;
  egfr_info?: {
    egfr: number;
    egfr_stage: string;
    message: string;
  };
  processed_values?: Array<{
    parameter: string;
    original_value: number;
    original_unit: string;
    converted_value: number;
    converted_unit: string;
    date: string;
    warnings: string[];
  }>;
  errors?: string[];
}

export interface HealthResponse {
  status: string;
  timestamp: number;
  version: string;
  models_loaded: number;
  system_info?: Record<string, any>;
}

export interface PerformanceMetrics {
  study_info: {
    title: string;
    description: string;
    datasets: string[];
    time_horizons: string[];
    sample_size: string;
    validation_method: string;
  };
  metrics: Record<string, any>;
  key_findings: string[];
  clinical_interpretation: Record<string, string>;
}

export interface ClinicalDisclaimer {
  title: string;
  last_updated: string;
  sections: Record<string, {
    title: string;
    content: string[];
  }>;
  contact_info: Record<string, string>;
  regulatory_note: string;
}

// API Service Class
export class ApiService {
  /**
   * Predict CKD risk for a patient
   */
  static async predictRisk(request: PredictionRequest): Promise<PredictionResult> {
    try {
      const response: AxiosResponse<PredictionResult> = await apiClient.post('/predict', request);
      return response.data;
    } catch (error: any) {
      console.error('Prediction API Error:', error);
      
      // Extract meaningful error message from API response
      if (error.response?.data?.message?.message) {
        // Handle nested error structure: {message: {message: "actual error"}}
        throw new Error(error.response.data.message.message);
      } else if (error.response?.data?.message) {
        // Handle simple message
        throw new Error(error.response.data.message);
      } else if (error.response?.data?.error) {
        throw new Error(error.response.data.error);
      } else if (error.message) {
        throw new Error(error.message);
      } else {
        throw new Error('Failed to generate prediction. Please check your input and try again.');
      }
    }
  }

  /**
   * Validate patient input without generating predictions
   */
  static async validateInput(request: PredictionRequest): Promise<ValidationResponse> {
    try {
      const response: AxiosResponse<ValidationResponse> = await apiClient.post('/predict/validate', request);
      return response.data;
    } catch (error) {
      console.error('Validation API Error:', error);
      throw error;
    }
  }

  /**
   * Check API health status
   */
  static async checkHealth(): Promise<HealthResponse> {
    try {
      const response: AxiosResponse<HealthResponse> = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health Check API Error:', error);
      throw error;
    }
  }

  /**
   * Get detailed health status
   */
  static async getDetailedHealth(): Promise<HealthResponse> {
    try {
      const response: AxiosResponse<HealthResponse> = await apiClient.get('/health/detailed');
      return response.data;
    } catch (error) {
      console.error('Detailed Health API Error:', error);
      throw error;
    }
  }

  /**
   * Get model status information
   */
  static async getModelStatus(): Promise<Record<string, any>> {
    try {
      const response: AxiosResponse<Record<string, any>> = await apiClient.get('/health/models');
      return response.data;
    } catch (error) {
      console.error('Model Status API Error:', error);
      throw error;
    }
  }

  /**
   * Get performance metrics
   */
  static async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    try {
      const response: AxiosResponse<PerformanceMetrics> = await apiClient.get('/info/performance');
      return response.data;
    } catch (error) {
      console.error('Performance Metrics API Error:', error);
      throw error;
    }
  }

  /**
   * Get clinical disclaimer
   */
  static async getClinicalDisclaimer(): Promise<ClinicalDisclaimer> {
    try {
      const response: AxiosResponse<ClinicalDisclaimer> = await apiClient.get('/info/disclaimer');
      return response.data;
    } catch (error) {
      console.error('Clinical Disclaimer API Error:', error);
      throw error;
    }
  }

  /**
   * Get clinical benchmarks
   */
  static async getClinicalBenchmarks(): Promise<Record<string, any>> {
    try {
      const response: AxiosResponse<Record<string, any>> = await apiClient.get('/info/clinical-benchmarks');
      return response.data;
    } catch (error) {
      console.error('Clinical Benchmarks API Error:', error);
      throw error;
    }
  }

  /**
   * Get unit conversion information
   */
  static async getUnitConversions(): Promise<Record<string, any>> {
    try {
      const response: AxiosResponse<Record<string, any>> = await apiClient.get('/info/units');
      return response.data;
    } catch (error) {
      console.error('Unit Conversions API Error:', error);
      throw error;
    }
  }
}

// Utility functions
export const formatDate = (date: Date): string => {
  return date.toISOString().split('T')[0];
};

export const parseApiError = (error: any): string => {
  if (error.response?.data?.message) {
    return error.response.data.message;
  } else if (error.message) {
    return error.message;
  } else {
    return 'An unexpected error occurred. Please try again.';
  }
};

// Export axios instance for direct use if needed
export { apiClient };

export default ApiService;